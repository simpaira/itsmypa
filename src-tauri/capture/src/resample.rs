//! Streaming windowed-sinc resampler for mono speech audio.
//!
//! Capture devices run at whatever rate they like (48 kHz built-in mic,
//! 24/16 kHz Bluetooth headsets, 44.1 kHz speakers) and the rate can change
//! mid-meeting when the user switches devices. Every track is converted to a
//! fixed 16 kHz as it's written, so one WAV file can span device switches and
//! the models get the rate they expect. The low-pass is built into the kernel,
//! so downsampling doesn't alias high frequencies into the speech band.

/// Kernel half-width, in zero crossings of the low-pass sinc.
const ZERO_CROSSINGS: f64 = 12.0;

pub struct Resampler {
    step: f64,         // input samples advanced per output sample
    cutoff: f64,       // low-pass cutoff, cycles per input sample (≤ 0.5)
    half_width: usize, // kernel half-width, in input samples
    buf: Vec<f32>,     // input history; buf[0] is absolute input index `base`
    base: i64,
    t: f64, // absolute input position of the next output sample
    emitted: usize,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let ratio = out_rate as f64 / in_rate as f64;
        // 0.45 rather than 0.5 leaves a transition band below Nyquist.
        let cutoff = 0.45 * ratio.min(1.0);
        let half_width = (ZERO_CROSSINGS / (2.0 * cutoff)).ceil() as usize;
        Self {
            step: in_rate as f64 / out_rate as f64,
            cutoff,
            half_width,
            buf: Vec::new(),
            base: 0,
            t: 0.0,
            emitted: 0,
        }
    }

    fn kernel(&self, x: f64) -> f64 {
        let w = x / self.half_width as f64;
        if w.abs() >= 1.0 {
            return 0.0;
        }
        let arg = 2.0 * self.cutoff * x;
        let sinc = if arg.abs() < 1e-9 {
            1.0
        } else {
            (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
        };
        // Blackman window over [-half_width, half_width].
        let a = std::f64::consts::PI * (w + 1.0);
        let window = 0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos();
        2.0 * self.cutoff * sinc * window
    }

    fn sample(&self, i: i64) -> f64 {
        if i < self.base {
            return 0.0; // before the stream started
        }
        self.buf.get((i - self.base) as usize).map_or(0.0, |&s| s as f64)
    }

    /// Feed input; returns every output sample that can be computed so far.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(input);
        let hw = self.half_width as i64;
        let end = self.base + self.buf.len() as i64; // one past last input index
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 1);
        while (self.t.floor() as i64) + hw < end {
            let center = self.t.floor() as i64;
            let mut acc = 0.0;
            for i in (center - hw + 1)..=(center + hw) {
                acc += self.sample(i) * self.kernel(self.t - i as f64);
            }
            out.push(acc as f32);
            self.t += self.step;
        }
        self.emitted += out.len();
        // Drop history no future output can reach.
        let keep_from = (self.t.floor() as i64 - hw).max(self.base);
        let drop = (keep_from - self.base) as usize;
        if drop > 0 {
            self.buf.drain(..drop);
            self.base = keep_from;
        }
        out
    }

    /// Emit the tail still waiting on look-ahead samples (end of stream).
    pub fn flush(&mut self) -> Vec<f32> {
        let total_in = self.base + self.buf.len() as i64;
        let before = self.emitted;
        let mut out = self.process(&vec![0.0; self.half_width]);
        // Keep only output samples whose position lies within the real input.
        let real = (total_in as f64 / self.step).ceil() as usize;
        out.truncate(real.saturating_sub(before));
        self.emitted = before + out.len();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        let n = (rate as f64 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32 * 0.5)
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn run(r: &mut Resampler, input: &[f32], block: usize) -> Vec<f32> {
        let mut out: Vec<f32> = input.chunks(block).flat_map(|c| r.process(c)).collect();
        out.extend(r.flush());
        out
    }

    #[test]
    fn output_length_matches_rate_ratio() {
        for (inr, outr) in [(48_000, 16_000), (44_100, 16_000), (24_000, 16_000), (8_000, 16_000)] {
            let input = tone(440.0, inr, 1.0);
            let out = run(&mut Resampler::new(inr, outr), &input, 480);
            let want = outr as i64;
            assert!((out.len() as i64 - want).abs() <= 2, "{inr}->{outr}: got {}", out.len());
        }
    }

    #[test]
    fn speech_band_passes_and_aliases_are_removed() {
        // 1 kHz is speech-band: keeps its level. 12 kHz is above the 8 kHz
        // output Nyquist: must be filtered out, not folded down to 4 kHz.
        let pass = run(&mut Resampler::new(48_000, 16_000), &tone(1_000.0, 48_000, 1.0), 512);
        let stop = run(&mut Resampler::new(48_000, 16_000), &tone(12_000.0, 48_000, 1.0), 512);
        let mid = |v: &[f32]| v[2_000..14_000].to_vec();
        assert!((rms(&mid(&pass)) - 0.5 / 2f32.sqrt()).abs() < 0.01, "passband rms {}", rms(&mid(&pass)));
        assert!(rms(&mid(&stop)) < 0.005, "alias rms {}", rms(&mid(&stop)));
    }

    #[test]
    fn block_size_does_not_change_output() {
        let input = tone(700.0, 44_100, 0.5);
        let a = run(&mut Resampler::new(44_100, 16_000), &input, 37);
        let b = run(&mut Resampler::new(44_100, 16_000), &input, 4_096);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-6));
    }
}
