# Installing ItsMyPA

**Requirements:** Apple Silicon Mac (M1 or newer), macOS 14.6 Sonoma or newer,
16 GB RAM recommended, ~4 GB free disk (the app is 14 MB; its AI models are
~3.2 GB).

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

1. **Model download (one time):** ItsMyPA downloads its AI models (~2.6 GB) to
   your app-data folder. On a typical connection this takes a few minutes —
   the app shows progress, and every file is checked against a known
   fingerprint. After this, ItsMyPA works fully offline.
2. **Two permissions** — the welcome screen asks for both; click **Allow**
   when macOS asks:
   - **Microphone** — so ItsMyPA can hear you.
   - **System Audio Recording** — to capture the *computer's* audio (the other
     people in your Zoom/Meet/Teams call). This is audio only — ItsMyPA never
     sees your screen. Without it, recording still works but is mic-only.

## Troubleshooting

- **"ItsMyPA is damaged and can't be opened"** — run
  `xattr -cr /Applications/ItsMyPA.app` and launch again.
- **No "Open Anyway" button** — launch the app once first (get blocked), then
  check System Settings → Privacy & Security again.
- **Other participants aren't in the transcript** — allow ItsMyPA under
  System Settings → Privacy & Security → Screen & System Audio Recording →
  **System Audio Recording Only**, then start a new recording.
- **Asked for permissions again after updating** — expected for now: until the
  app is signed, macOS treats each new version as a new app. Allow again once.
- **The app quit during a meeting** — reopen it: the home screen offers to
  recover the unfinished recording.
- **Model download fails** — check your connection and relaunch; downloads
  resume where they stopped. Corporate networks that block Hugging Face can be
  the cause — the app automatically falls back to a mirror.

## Uninstall

Delete `/Applications/ItsMyPA.app` and (optionally) your data at
`~/Library/Application Support/ItsMyPA`.
