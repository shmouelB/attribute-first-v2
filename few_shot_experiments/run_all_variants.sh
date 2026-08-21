#!/usr/bin/env bash
# Thin compatibility facade for the controlled sixteen-cell campaign.
# Usage: cd few_shot_experiments &&
#   RUN_ID=<id> bash run_all_variants.sh [--max-workers N]
# Offline checks are documented by attribute_first.campaign.cli.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$SCRIPT_DIR/../.venv/bin/python" ]]; then
        PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi
exec "$PYTHON_BIN" -B -m attribute_first.campaign.cli "$@"
