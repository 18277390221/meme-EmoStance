#!/usr/bin/env bash
set -euo pipefail
version_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$version_dir/runtime_source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python_executable="${MEME_PYTHON:-/home/aida102/emoji/.venv/bin/python}"
exec "$python_executable" -m emostance_meme.paper_pipeline chat --config "$version_dir/inference_config.json" "$@"
