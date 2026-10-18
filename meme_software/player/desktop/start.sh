#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONUTF8=1
exec /usr/bin/python3 app.py --port 8784 --end-silence 0.7 "$@"
