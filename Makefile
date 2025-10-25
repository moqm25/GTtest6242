PYTHON := python
STREAMLIT := streamlit

.PHONY: setup env install data ingest build score ml reco app-smoke test format lint audit clean-data

# Full one-shot setup: env + deps + data
setup: env install data
	@echo "==> Setup complete. Try: make ingest && make build && make score && make app-smoke"

# Create local Python via pyenv and install deps
env:
	@echo "==> Bootstrapping environment..."
	@bash scripts/bootstrap.sh

# Install/refresh requirements into current env
install:
	pip install --upgrade pip wheel setuptools
	pip install -r requirements.txt

# Download & stage USDA CSVs into DATA/
data:
	@echo "==> Downloading USDA data into DATA/ ..."
	@USE_RICH=1 python scripts/setup_data.py

ingest:
	USE_RICH=1 $(PYTHON) -m src.ingest_usda

build:
	USE_RICH=1 $(PYTHON) -m src.build_dataset

score:
	USE_RICH=1 $(PYTHON) -m src.scoring

ml:
	USE_RICH=1 $(PYTHON) -m src.ml.train

reco:
	USE_RICH=1 $(PYTHON) -m src.reco --rebuild

app-smoke:
	STREAMLIT_SERVER_HEADLESS=true streamlit run app/app.py --server.headless true --server.port 8501

test:
	pytest

format:
	isort .
	black .
	ruff check .

audit:
	python scripts/audit_repo.py

# Optional: nuke staged CSVs and processed outputs
clean-data:
	@echo "==> Removing staged CSVs and processed files..."
	@rm -f DATA/*.csv
	@rm -rf DATA/processed/*
