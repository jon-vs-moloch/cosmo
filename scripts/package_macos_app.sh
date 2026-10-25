#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="$ROOT/dist/Cosmo.app"
PYTHON_BIN="${COSMO_PYTHON:-$(command -v python3 || true)}"
DRY_RUN=0

usage() {
  print "Usage: scripts/package_macos_app.sh [--output PATH] [--python PATH] [--dry-run]"
}

while (( $# )); do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) print -u2 "ERROR: unknown option: $1"; usage; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || { print -u2 "ERROR: Cosmo.app can only be built on macOS."; exit 2; }
for tool in swiftc sips iconutil plutil rsync; do
  command -v "$tool" >/dev/null || { print -u2 "ERROR: required macOS tool is missing: $tool"; exit 2; }
done
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { print -u2 "ERROR: no executable Python found; pass --python /path/to/python3."; exit 2; }
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  print -u2 "ERROR: Cosmo requires Python 3.11 or newer; choose it with --python or COSMO_PYTHON."
  exit 2
}

VERSION="$($PYTHON_BIN -c "import sys; sys.path.insert(0, '$ROOT/src'); import cosmo; print(cosmo.__version__)")"
print "Cosmo.app version $VERSION"
print "Runtime strategy: source-only bundle; launcher requires a user-installed Python >=3.11 at runtime."
print "Credential boundary: no provider credential, home-directory file, or environment secret is copied into the app."
if (( DRY_RUN )); then
  print "Dry run passed. Would write: $OUTPUT"
  exit 0
fi

[[ ! -e "$OUTPUT" ]] || { print -u2 "ERROR: output already exists: $OUTPUT (choose --output; refusing to overwrite)."; exit 2; }
APP="$OUTPUT"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/python" "$APP/Contents/Resources/benchmarks"

sed "s/@COSMO_VERSION@/$VERSION/g" "$ROOT/resources/Info.plist.in" > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist" >/dev/null
swiftc "$ROOT/resources/CosmoLauncher.swift" -framework AppKit -o "$APP/Contents/MacOS/Cosmo"
"$ROOT/scripts/generate_macos_icon.sh" "$APP/Contents/Resources/Cosmo.icns"

rsync -a --exclude '__pycache__' --exclude '*.py[co]' \
  "$ROOT/src/cosmo/" "$APP/Contents/Resources/python/cosmo/"
ditto "$ROOT/benchmarks" "$APP/Contents/Resources/benchmarks"
cp "$ROOT/pyproject.toml" "$APP/Contents/Resources/pyproject.toml"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$APP/Contents/Resources/"

cat > "$APP/Contents/Resources/launch_cosmo.command" <<'COMMAND'
#!/bin/zsh
set -o pipefail
APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$HOME/Library/Logs/Cosmo"
LOG_FILE="$LOG_DIR/cosmo.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

print "Cosmo experimental harness"
print "Log: $LOG_FILE"
print "Cosmo does not include provider credentials; LM Studio remains operator-owned at runtime."

PYTHON_BIN="${COSMO_PYTHON:-$(command -v python3 || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  print -u2 "ERROR: Cosmo.app needs a user-installed Python 3.11+ (set COSMO_PYTHON to override)."
  print -u2 "No credentials were read or copied. Install Python, then launch Cosmo again."
  exit 2
fi
export PYTHONPATH="$APP_ROOT/Resources/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if (( $# == 0 )); then
  print "No command supplied; showing the Cosmo command guide."
  "$PYTHON_BIN" -m cosmo.cli --help
  print ""
  print "Example: verify the local LM Studio capability:"
  print "  $PYTHON_BIN -m cosmo.cli doctor --model <loaded-instance> --probe"
  print "  $PYTHON_BIN -m cosmo.cli atomic --help"
  print "  $PYTHON_BIN -m cosmo.cli run --model <loaded-instance>"
  print "    --model-dir /path/to/model --chat-template /path/to/chat_template.jinja"
  print "    --benchmark $APP_ROOT/Resources/benchmarks/iterative-self-verification/v1/cases.jsonl"
  print "    --arms instant_recursive"
  exit 0
fi
"$PYTHON_BIN" -m cosmo.cli "$@"
exit_status=$?
print "Cosmo exited with status $exit_status."
exit $exit_status
COMMAND
chmod 755 "$APP/Contents/Resources/launch_cosmo.command"

"$ROOT/scripts/check_macos_app.sh" "$APP"
print "Packaged $APP"
