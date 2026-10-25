#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The build renders from a committed 1024x1024 PNG. resources/cosmo-icon.svg is
# the editable source; after changing it, re-render the PNG on macOS with
#   sips -s format png resources/cosmo-icon.svg --out resources/cosmo-icon.png
# and commit both. SVG rendering through sips depends on system services that
# are not available in every (sandboxed/headless) build environment, so the
# packaging path only ever resizes the PNG.
SOURCE="$ROOT/resources/cosmo-icon.png"
OUTPUT="${1:-$ROOT/resources/Cosmo.icns}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/cosmo-icon.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

command -v sips >/dev/null || { print -u2 "ERROR: macOS sips is required to render the Cosmo icon."; exit 2; }
command -v iconutil >/dev/null || { print -u2 "ERROR: macOS iconutil is required to create an .icns file."; exit 2; }

[[ -f "$SOURCE" ]] || { print -u2 "ERROR: icon source is missing: $SOURCE"; exit 2; }
mkdir -p "$WORK/Cosmo.iconset"
cp "$SOURCE" "$WORK/source.png"
for spec in "16 16x16" "32 16x16@2x" "32 32x32" "64 32x32@2x" "128 128x128" "256 128x128@2x" "256 256x256" "512 256x256@2x" "512 512x512" "1024 512x512@2x"; do
  parts=(${=spec})
  sips -z "$parts[1]" "$parts[1]" "$WORK/source.png" --out "$WORK/Cosmo.iconset/icon_$parts[2].png" >/dev/null
done
mkdir -p "$(dirname "$OUTPUT")"
iconutil -c icns "$WORK/Cosmo.iconset" -o "$OUTPUT"
print "Generated $OUTPUT"
