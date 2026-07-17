# Installing ItsMyPA

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4), macOS 13 Ventura or newer,
16 GB RAM recommended, ~6 GB free disk (app + AI models).

## Option A — Homebrew (easiest, no security dialogs)

```bash
brew install --cask simpaira/tap/itsmypa
```

Done. Skip to [First launch](#first-launch).

## Option B — DMG

1. **Download** the latest `ItsMyPA.dmg` from the
   [Releases page](https://github.com/simpaira/itsmypa/releases/latest).
2. **Open the DMG** and drag **ItsMyPA** into **Applications**.
3. **Launch it.** macOS will block it with *"ItsMyPA" can't be opened because
   Apple cannot verify it is free of malware.*

   > **Why this happens:** we haven't paid Apple's $99/year code-signing fee.
   > The app isn't broken and isn't malware — the full source code is public in
   > this repo, which is more than a certificate proves. Click **Done** (don't
   > move it to Trash) and continue:

4. Open **System Settings → Privacy & Security**, scroll down to the Security
   section — you'll see *"ItsMyPA" was blocked to protect your Mac.*
5. Click **Open Anyway**, then confirm with **Open Anyway** again (macOS asks
   for your password or Touch ID).

   **Terminal alternative** (does the same thing in one line):

   ```bash
   xattr -cr /Applications/ItsMyPA.app
   ```

## First launch

1. **Model download (one time):** ItsMyPA downloads its AI models (~5 GB) to
   your app-data folder. On a typical connection this takes a few minutes —
   the app shows progress. After this, ItsMyPA works fully offline.
2. **Microphone permission:** macOS asks on your first recording — click Allow.
3. **Screen Recording permission:** needed to capture the *computer's* audio
   (the other people in your Zoom/Meet/Teams call). ItsMyPA never captures your
   screen content — macOS just gates system-audio capture behind this
   permission. Without it, recording still works but is mic-only.

## Troubleshooting

- **"ItsMyPA is damaged and can't be opened"** — run
  `xattr -cr /Applications/ItsMyPA.app` and launch again.
- **No "Open Anyway" button** — launch the app once first (get blocked), then
  check System Settings → Privacy & Security again.
- **Other participants aren't in the transcript** — grant Screen Recording
  permission (System Settings → Privacy & Security → Screen & System Audio
  Recording), then restart the recording.
- **Model download fails** — check your connection and relaunch; downloads
  resume. Corporate networks that block Hugging Face can be the cause — the app
  automatically falls back to a mirror.

## Uninstall

Delete `/Applications/ItsMyPA.app` and (optionally) your data at
`~/Library/Application Support/ItsMyPA`.
