#!/usr/bin/env bash
set -euo pipefail
EMOSTANCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$EMOSTANCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -n ${EMOSTANCE_PYTHON:-} ]]; then
    exec "$EMOSTANCE_PYTHON" -m emostance_meme "$@"
elif [[ -x /home/aida102/emoji/.venv/bin/python ]]; then
    exec /home/aida102/emoji/.venv/bin/python -m emostance_meme "$@"
else
    exec python3 -m emostance_meme "$@"
fi
