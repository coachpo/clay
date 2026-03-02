#!/usr/bin/env bash
set -euo pipefail

# Run from repository root so module imports resolve consistently.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

exec python -m app.main "$@"
