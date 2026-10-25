#!/bin/zsh
set -euo pipefail

APP="${1:?Usage: scripts/check_macos_app.sh /path/to/Cosmo.app}"
INFO="$APP/Contents/Info.plist"
[[ -d "$APP" ]] || { print -u2 "ERROR: app bundle does not exist: $APP"; exit 2; }
for required_path in "$INFO" "$APP/Contents/MacOS/Cosmo" "$APP/Contents/Resources/Cosmo.icns" "$APP/Contents/Resources/launch_cosmo.command" "$APP/Contents/Resources/python/cosmo/cli.py" "$APP/Contents/Resources/LICENSE" "$APP/Contents/Resources/NOTICE"; do
  [[ -e "$required_path" ]] || { print -u2 "ERROR: required bundle file is missing: $required_path"; exit 2; }
done
plutil -lint "$INFO" >/dev/null
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$INFO")" == "com.owlandkestrel.cosmo" ]] || { print -u2 "ERROR: unexpected bundle identifier"; exit 2; }
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIconFile' "$INFO")" == "Cosmo.icns" ]] || { print -u2 "ERROR: unexpected icon metadata"; exit 2; }
[[ -x "$APP/Contents/MacOS/Cosmo" && -x "$APP/Contents/Resources/launch_cosmo.command" ]] || { print -u2 "ERROR: launcher is not executable"; exit 2; }

# A package must not accidentally carry credential-shaped artifacts.
if find "$APP" -type f \( -name 'auth.json' -o -name '.env' -o -name 'credentials*' \) -print -quit | grep -q .; then
  print -u2 "ERROR: credential-shaped artifact found in bundle"
  exit 2
fi
if find "$APP/Contents/Resources/python" \( -type d -name '__pycache__' -o -type f -name '*.py[co]' \) -print -quit | grep -q .; then
  print -u2 "ERROR: generated Python cache leaked into the app bundle"
  exit 2
fi

if command -v codesign >/dev/null; then
  codesign --display --verbose=1 "$APP" 2>&1 | grep -q 'Identifier=' || print "NOTE: unsigned development bundle (expected)."
fi
print "Bundle checks passed: $APP"
