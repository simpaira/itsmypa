// itsmypa-audio — native audio capture helper for ItsMyPA.
//
//   itsmypa-audio <output.wav>              capture system audio (ScreenCaptureKit)
//   itsmypa-audio --mic <output.wav>        capture the default microphone (AVAudioEngine)
//   itsmypa-audio --mic-status              print mic TCC state (granted/denied/undetermined), no prompt
//   itsmypa-audio --sys-status              print Screen Recording TCC state (granted/denied), no prompt
//
// The mic is captured HERE, natively, instead of via the webview's
// getUserMedia: WebKit engages macOS voice processing for its mic capture,
// which mutes all other system audio — a meeting recorder must leave the
// meeting audible. AVAudioEngine's plain input tap does no voice processing.
//
// Captures until SIGINT/SIGTERM, then finalizes the WAV. SIGUSR1 pauses the
// mic capture, SIGUSR2 resumes it. While capturing the mic, RMS level lines
// ("L 0.42") are printed to stderr ~10×/s so the UI can draw a live waveform.
//
// Teardown discipline matters: streams MUST be stopped before the process
// exits. Dying mid-capture leaves dangling audio-HAL clients in coreaudiod,
// and repeated abrupt deaths can wedge system audio entirely.

import AVFoundation
import ScreenCaptureKit

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

func emit(_ msg: String) {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
}

// ── System audio (ScreenCaptureKit) ─────────────────────────────────────────
final class Recorder: NSObject, SCStreamOutput, @unchecked Sendable {
    private let outputURL: URL
    private var audioFile: AVAudioFile?
    private var stream: SCStream?
    private var stopping = false
    private let lock = NSLock()

    init(outputURL: URL) { self.outputURL = outputURL }

    // NSLock.lock() is unavailable directly in async contexts; funnel all
    // critical sections through this synchronous helper instead.
    private func sync<T>(_ body: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        return body()
    }

    func start() async {
        let content: SCShareableContent
        do {
            // Throws if Screen Recording permission hasn't been granted.
            content = try await SCShareableContent.excludingDesktopWindows(false,
                                                                           onScreenWindowsOnly: false)
        } catch {
            fail("permission-needed: \(error.localizedDescription)")
        }
        guard let display = content.displays.first else { fail("no-display") }

        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        // A display stream still needs a video size; keep it minimal, we ignore frames.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: config, delegate: nil)
        do {
            try stream.addStreamOutput(self, type: .audio,
                                       sampleHandlerQueue: DispatchQueue(label: "itsmypa.audio"))
            try await stream.startCapture()
        } catch {
            fail("capture-failed: \(error.localizedDescription)")
        }

        let stopRequestedDuringStartup: Bool = sync {
            if !stopping { self.stream = stream }
            return stopping
        }
        // A stop signal can land while startCapture() is still in flight —
        // tear the fresh stream down properly instead of letting exit() reap it.
        if stopRequestedDuringStartup {
            try? await stream.stopCapture()
            exit(0)
        }
        emit("capturing")
    }

    // Stops the capture stream and finalizes the WAV. Safe to call repeatedly.
    func stopAndFinalize() async {
        let s: SCStream? = sync {
            stopping = true
            let s = stream
            stream = nil
            return s
        }
        if let s { try? await s.stopCapture() }
        sync { audioFile = nil }   // closes/finalizes the WAV
    }

    // MARK: SCStreamOutput
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid,
              let fmtDesc = sampleBuffer.formatDescription,
              let asbd = fmtDesc.audioStreamBasicDescription else { return }

        lock.lock(); defer { lock.unlock() }
        guard !stopping else { return }

        if audioFile == nil {
            var streamDesc = asbd
            guard let format = AVAudioFormat(streamDescription: &streamDesc) else { return }
            // Write a standard 16-bit PCM WAV so downstream ffmpeg/pydub read it cleanly.
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: format.sampleRate,
                AVNumberOfChannelsKey: format.channelCount,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
            ]
            audioFile = try? AVAudioFile(forWriting: outputURL, settings: settings)
        }
        guard let audioFile else { return }

        var streamDesc = asbd
        guard let format = AVAudioFormat(streamDescription: &streamDesc) else { return }
        do {
            try sampleBuffer.withAudioBufferList { abl, _ in
                guard let pcm = AVAudioPCMBuffer(pcmFormat: format,
                                                 bufferListNoCopy: abl.unsafePointer) else { return }
                try? audioFile.write(from: pcm)
            }
        } catch { /* drop this buffer */ }
    }
}

// ── Microphone (AVAudioEngine input tap; NO voice processing) ───────────────
final class MicRecorder: @unchecked Sendable {
    private let outputURL: URL
    private let engine = AVAudioEngine()
    private var audioFile: AVAudioFile?
    private var stopping = false
    private var paused = false
    private var lastLevel = Date.distantPast
    private let lock = NSLock()

    init(outputURL: URL) { self.outputURL = outputURL }

    private func sync<T>(_ body: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        return body()
    }

    func start() {
        // Trigger the system mic prompt (attributed to the responsible app,
        // which carries NSMicrophoneUsageDescription) and wait for the answer.
        let sem = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .audio) { g in granted = g; sem.signal() }
        sem.wait()
        if !granted { fail("permission-needed: microphone access declined") }

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { fail("no-input-device") }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do { audioFile = try AVAudioFile(forWriting: outputURL, settings: settings) }
        catch { fail("file-failed: \(error.localizedDescription)") }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buf, _ in
            self?.handle(buf)
        }
        do { try engine.start() }
        catch { fail("mic-failed: \(error.localizedDescription)") }
        emit("capturing")
    }

    func setPaused(_ p: Bool) { sync { paused = p } }

    func stopAndFinalize() async {
        sync { stopping = true }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        sync { audioFile = nil }   // closes/finalizes the WAV
    }

    private func handle(_ buf: AVAudioPCMBuffer) {
        lock.lock(); defer { lock.unlock() }
        guard !stopping, !paused, let audioFile else { return }
        try? audioFile.write(from: buf)

        // RMS level for the UI's live waveform, throttled to ~10 lines/s.
        let now = Date()
        guard now.timeIntervalSince(lastLevel) > 0.1,
              let data = buf.floatChannelData, buf.frameLength > 0 else { return }
        lastLevel = now
        var sum: Float = 0
        let n = Int(buf.frameLength)
        for i in 0..<n { let s = data[0][i]; sum += s * s }
        let rms = (sum / Float(n)).squareRoot()
        emit(String(format: "L %.3f", min(rms * 4, 1)))   // scaled for display
    }
}

// ── entry ──
var micMode = false
var statusMode = false
var sysStatusMode = false
var outPath: String? = nil
for arg in CommandLine.arguments.dropFirst() {
    switch arg {
    case "--mic": micMode = true
    case "--mic-status": statusMode = true
    case "--sys-status": sysStatusMode = true
    default: outPath = arg
    }
}

if statusMode {
    // Report without prompting, so the app can check state silently at launch.
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: print("granted")
    case .denied, .restricted: print("denied")
    default: print("undetermined")
    }
    exit(0)
}

if sysStatusMode {
    // Screen Recording state without prompting or starting a capture.
    print(CGPreflightScreenCaptureAccess() ? "granted" : "denied")
    exit(0)
}

guard let outPath else { fail("usage: itsmypa-audio [--mic] <output.wav>") }
let outURL = URL(fileURLWithPath: outPath)

let sysRecorder: Recorder? = micMode ? nil : Recorder(outputURL: outURL)
let micRecorder: MicRecorder? = micMode ? MicRecorder(outputURL: outURL) : nil

nonisolated(unsafe) var exiting = false
func stopAndExit() {
    if exiting { return }
    exiting = true
    Task {
        await sysRecorder?.stopAndFinalize()
        await micRecorder?.stopAndFinalize()
        exit(0)
    }
    // Hard deadline: never hang past teardown, but give the streams a real
    // chance to stop — exit(0) from a raw signal handler skipped it entirely.
    DispatchQueue.main.asyncAfter(deadline: .now() + 3) { exit(0) }
}

// Raw C signal handlers can't safely run Swift/async teardown; route the
// signals through dispatch sources on the main queue instead.
for sig in [SIGINT, SIGTERM, SIGUSR1, SIGUSR2] { signal(sig, SIG_IGN) }
let sigintSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigintSrc.setEventHandler { stopAndExit() }
sigintSrc.resume()
let sigtermSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
sigtermSrc.setEventHandler { stopAndExit() }
sigtermSrc.resume()
let pauseSrc = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
pauseSrc.setEventHandler { micRecorder?.setPaused(true) }
pauseSrc.resume()
let resumeSrc = DispatchSource.makeSignalSource(signal: SIGUSR2, queue: .main)
resumeSrc.setEventHandler { micRecorder?.setPaused(false) }
resumeSrc.resume()

// Orphan watchdog: if the parent server dies without stopping us (app quit,
// crash), don't keep capturing forever — stop cleanly and exit.
let watchdog = DispatchSource.makeTimerSource(queue: .main)
watchdog.schedule(deadline: .now() + 2, repeating: 2)
watchdog.setEventHandler { if getppid() == 1 { stopAndExit() } }
watchdog.resume()

if let sysRecorder {
    Task { await sysRecorder.start() }
} else {
    micRecorder?.start()
}
RunLoop.main.run()
