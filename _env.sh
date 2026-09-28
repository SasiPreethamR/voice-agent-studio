# Sourced by the start_*.sh scripts: loads .env and picks the Python binary.
# systemd-run does not inherit the calling shell's environment, so scripts pass
# the values they need with --setenv / -E.
set -a
[ -f "$SCRIPT_DIR/.env" ] && . "$SCRIPT_DIR/.env"
set +a
if [ -z "$PYTHON" ]; then
  if [ -x /usr/bin/python3 ]; then PYTHON=/usr/bin/python3; else PYTHON="$(command -v python3)"; fi
fi
mkdir -p "$SCRIPT_DIR/logs"
