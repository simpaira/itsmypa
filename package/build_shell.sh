#!/usr/bin/env bash
# Build the full ItsMyPA desktop app: Python engine bundle → Tauri native shell
# → signed-later .app → drag-to-Applications .dmg.
#
#   ./package/build_shell.sh
#
# We build the .app with Tauri, then wrap the DMG ourselves with hdiutil —
# Tauri's built-in DMG step relies on Finder AppleScript that needs a GUI login
# session and fails in headless/CI contexts.
set -e
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

echo "── 1/3  Building the Python engine bundle ──"
./package/build_server.sh

echo "── 2/3  Building the Tauri native shell ──"
cargo tauri build

APP="src-tauri/target/release/bundle/macos/ItsMyPA.app"
[ -d "$APP" ] || { echo "❌ $APP not found"; exit 1; }

# Tauri ad-hoc signs the app as "itsmypa-<content hash>", which macOS can't map
# back to the installed bundle — System Settings then shows a generic icon and
# a new permission row per build. Re-sign with the real bundle identifier so
# LaunchServices can at least resolve the app's name and icon. (True permission
# persistence across builds still needs a Developer ID certificate.)
codesign --force --sign - --identifier com.itsmypa.desktop "$APP"

echo "── 3/3  Wrapping the DMG ──"
mkdir -p dist
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f dist/ItsMyPA.dmg
hdiutil create -volname "ItsMyPA" -srcfolder "$STAGE" -ov -format UDZO dist/ItsMyPA.dmg >/dev/null
rm -rf "$STAGE"

echo
echo "✅ Done:"
du -sh "$APP" dist/ItsMyPA.dmg
echo
echo "Reminder: unsigned. For distribution, codesign + notarize the .app before"
echo "wrapping the DMG (needs an Apple Developer ID)."
