#!/usr/bin/env bash
# Public teacher snapshot -> full nonzero-seed replays -> teacher-seat BC index.
set -euo pipefail

BC_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BC_PROJECT_ROOT"
BC_DATA_ROOT="${1:-data/action_bc}"
BC_TOP_TEAMS="${2:-20}"
BC_GAMES_PER_TEACHER="${3:-5}"
BC_PYTHON="$BC_PROJECT_ROOT/.venv-bc-tools/bin/python"
BC_KAGGLE="$BC_PROJECT_ROOT/.venv-bc-tools/bin/kaggle"

if [[ ! -x "$BC_PYTHON" ]]; then
  python3 -m venv "$BC_PROJECT_ROOT/.venv-bc-tools"
fi
if ! "$BC_PYTHON" -c 'from importlib.metadata import version; assert version("kaggle") == "2.2.4"' >/dev/null 2>&1; then
  "$BC_PYTHON" -m pip install 'kaggle==2.2.4'
fi

export PYTHONPATH="$BC_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Apply the same seed=0 exclusion to old indexes before resuming. The downloader
# also excludes new zero-seed replays without charging the per-teacher quota.
"$BC_PYTHON" -m route_rl.replay_download filter-zero-seed --out "$BC_DATA_ROOT/public"
# Kaggle 2.2.4's login command exits 1 when credentials already exist. Probe
# usable authentication first so set -e does not stop a resumed download.
if ! "$BC_PYTHON" - <<'PY'
import sys
from route_rl.replay_download import kaggle_api

try:
    kaggle_api()
except RuntimeError:
    sys.exit(1)
print("Kaggle authentication is ready; continuing with saved credentials.")
PY
then
  "$BC_KAGGLE" auth login --no-launch-browser
fi
if [[ ! -f "$BC_DATA_ROOT/teachers.json" ]]; then
  "$BC_PYTHON" -m route_rl.replay_download discover \
    --top-teams "$BC_TOP_TEAMS" --out "$BC_DATA_ROOT/teachers.json"
fi
"$BC_PYTHON" -m route_rl.replay_download download \
  --teachers "$BC_DATA_ROOT/teachers.json" \
  --out "$BC_DATA_ROOT/public" --limit-per-teacher "$BC_GAMES_PER_TEACHER"
