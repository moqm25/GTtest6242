#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_VER="3.13.1"
VENV_NAME="cse6242-${PY_VER}"

cat <<EOF
==> Manual setup checklist

1) Install Python ${PY_VER} (recommended via pyenv):
   pyenv install ${PY_VER}
   pyenv virtualenv ${PY_VER} ${VENV_NAME}
   pyenv activate ${VENV_NAME}
   # alternatively: python${PY_VER%.*} -m venv .venv && source .venv/bin/activate

2) Upgrade tooling and install project requirements from ${PROJECT_ROOT}/requirements.txt:
   pip install --upgrade pip wheel setuptools
   pip install -r "${PROJECT_ROOT}/requirements.txt"

3) Download USDA FoodData Central CSVs into ${PROJECT_ROOT}/DATA/:
   USE_RICH=1 python "${PROJECT_ROOT}/scripts/setup_data.py"
   # rerun with --force to refresh the CSVs if needed

4) Run the pipeline targets after data is staged:
   make ingest
   make build
   make score
   make reco

Tips:
- macOS users can install pyenv with Homebrew: brew install pyenv pyenv-virtualenv
- Keep ${PROJECT_ROOT}/DATA/ under version control ignore; large CSVs are not committed.
- Activate ${VENV_NAME} (or your chosen virtualenv) before invoking make targets.

EOF
