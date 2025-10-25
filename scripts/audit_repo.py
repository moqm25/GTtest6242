#!/usr/bin/env python
"""Audit the nutrition scoring repository, auto-remediating minor gaps."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import subprocess

import numpy as np
import pandas as pd
import yaml

from ._util import format_table, run_command


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
LOG_PATH = DOCS_DIR / "AUDIT_LOG.txt"
REPORT_PATH = DOCS_DIR / "AUDIT_REPORT.md"


CANONICAL_NUTRIENTS = [
    "energy_kcal",
    "protein_g",
    "total_fat_g",
    "sat_fat_g",
    "mono_fat_g",
    "poly_fat_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "added_sugar_g",
    "sodium_mg",
    "potassium_mg",
]

REQUIRED_DIRECTORIES = [
    "data",
    "data/processed",
    "data/dictionaries",
    "src",
    "src/ml",
    "src/service",
    "app",
    "models",
    "notebooks",
    "tests",
    "docs",
    "configs",
    "scripts",
]

REQUIRED_FILES = [
    "Makefile",
    "README.md",
    ".pre-commit-config.yaml",
    "requirements.txt",
]

REQUIRED_CONFIGS = [
    "configs/paths.yml",
    "configs/scoring.yml",
    "configs/reco.yml",
]

REQUIRED_TESTS = [
    "tests/test_schema.py",
    "tests/test_ml.py",
    "tests/test_reco.py",
    "tests/test_golden.py",
]

MAKE_TARGETS = {
    "ingest": "python -m src.ingest_usda",
    "build": "python -m src.build_dataset",
    "score": "python -m src.scoring",
    "ml": "python -m src.ml.train",
    "reco": "python -m src.reco",
    "app-smoke": "streamlit run app/app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false --server.fileWatcherType none --server.enableCORS false --server.enableXsrfProtection false",
    "test": "pytest",
    "format": "isort . && black . && ruff check .",
    "audit": "python scripts/audit_repo.py",
}

REQUIRED_README_SECTIONS = [
    "## Overview",
    "## What's Included",
    "## Installation",
    "## Data Files",
    "## Quick Start",
    "## Health Score v1.0",
    "## Recommendations",
    "## App Usage",
    "## Tests",
    "## Design Notes & Limits",
    "## File Map",
    "## Troubleshooting",
]


class AuditState:
    """Hold audit findings and metrics."""

    def __init__(self) -> None:
        self.checks: List[Tuple[str, bool, str]] = []
        self.critical_failures: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.test_summary: str = "not run"
        self.app_smoke_status: str = "not run"
        self.modified_files: List[str] = []
        self.dataset_rows: int = 0
        self.required_nutrient_completeness: float = 0.0
        self.score_min: float = 0.0
        self.score_mean: float = 0.0
        self.score_max: float = 0.0
        self.reco_coverage: float = 0.0
        self.command_failures: List[str] = []

    def record(self, name: str, passed: bool, detail: str, critical: bool = False) -> None:
        self.checks.append((name, passed, detail))
        if critical and not passed:
            self.critical_failures.append(name)

    def log_modification(self, path: Path) -> None:
        self.modified_files.append(str(path.relative_to(ROOT)))


def append_log(message: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    formatted = f"[{timestamp}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(formatted + "\n")


def log_step(title: str) -> None:
    append_log("")
    append_log(f"=== {title} ===")


def discover_project_root() -> Path:
    append_log(f"Project root resolved to {ROOT}")
    return ROOT


def ensure_structure(state: AuditState) -> None:
    log_step("Structure Audit")
    missing_dirs: List[str] = []
    for rel in REQUIRED_DIRECTORIES:
        path = ROOT / rel
        if not path.exists():
            missing_dirs.append(rel)
            path.mkdir(parents=True, exist_ok=True)
            append_log(f"Created directory: {rel}")
            state.log_modification(path)
    missing_files: List[str] = []
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.exists():
            missing_files.append(rel)
            path.touch()
            append_log(f"Created empty file placeholder: {rel}")
            state.log_modification(path)
    detail_parts = []
    if missing_dirs:
        detail_parts.append(f"Created directories: {', '.join(missing_dirs)}")
    if missing_files:
        detail_parts.append(f"Created files: {', '.join(missing_files)}")
    detail = "; ".join(detail_parts) if detail_parts else "All required directories/files present"
    state.record("Structure", not bool(missing_dirs or missing_files), detail, critical=True)


def _makefile_has_target(content: str, target: str) -> bool:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(f"{target}:") or stripped.startswith(f"{target}::"):
            return True
    return False


def audit_makefile(state: AuditState) -> None:
    log_step("Makefile Audit")
    makefile_path = ROOT / "Makefile"
    content = makefile_path.read_text(encoding="utf-8")
    missing_targets = []
    for target, command in MAKE_TARGETS.items():
        if not _makefile_has_target(content, target):
            missing_targets.append(target)
            append_log(f"Appending missing make target: {target}")
            with makefile_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        f"""

.PHONY: {target}
{target}:
\t{command}
"""
                    )
                )
            state.log_modification(makefile_path)
            content += f"\n{target}:"
    if missing_targets:
        detail = f"Added targets: {', '.join(missing_targets)}"
    else:
        detail = "All required targets present"
    state.record("Makefile targets", not missing_targets, detail, critical=True)


def write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def ensure_configs(state: AuditState) -> None:
    log_step("Config Audit")
    defaults = {
        "configs/paths.yml": {
            "data_root": "DATA",
            "staging_dir": "data/raw",
            "processed_dir": "data/processed",
            "dictionaries_dir": "data/dictionaries",
            "docs_dir": "docs",
            "models_dir": "models",
        },
        "configs/scoring.yml": {
            "weights": {"macros": 0.45, "micros": 0.35, "penalties": 0.20},
            "daily_values": {
                "protein_g": 50.0,
                "fiber_g": 28.0,
                "potassium_mg": 4700.0,
                "added_sugar_g": 50.0,
                "sodium_mg": 2300.0,
                "sat_fat_g": 20.0,
                "energy_kcal": 2000.0,
            },
            "energy_density_reference": 275.0,
            "penalty_estimate_factor": 0.75,
            "grade_cutpoints": {"A": 85.0, "B": 75.0, "C": 65.0, "D": 50.0},
        },
        "configs/reco.yml": {
            "constraints": {
                "min_score_gain": 10.0,
                "max_kcal_delta": 100.0,
                "sodium_cap": 500.0,
                "form_match": True,
                "group_match": True,
            },
            "objective_weights": {"similarity": 0.6, "score_gain": 0.4},
            "neighbors": 50,
            "fallback_top_k": 3,
            "coverage_target": 0.9,
        },
    }
    missing = []
    for rel in REQUIRED_CONFIGS:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            write_yaml(path, defaults[rel])
            append_log(f"Wrote default config: {rel}")
            state.log_modification(path)
    detail = f"Created configs: {', '.join(missing)}" if missing else "All configs present"
    state.record("Configs", not missing, detail, critical=True)


def run_command_logged(state: AuditState, command: List[str], description: str, timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None, allow_failure: bool = True) -> int:
    append_log(f"Running command ({description}): {' '.join(command)}")
    try:
        completed = run_command(
            command,
            cwd=ROOT,
            timeout=timeout,
            env={**os.environ, **(env or {})},
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            append_log(f"STDOUT:\n{completed.stdout.strip()}")
        if completed.stderr:
            append_log(f"STDERR:\n{completed.stderr.strip()}")
        if completed.returncode != 0 and not allow_failure:
            state.command_failures.append(f"{description} (exit {completed.returncode})")
        return completed.returncode
    except subprocess.TimeoutExpired:  # type: ignore[name-defined]
        append_log(f"Command timed out after {timeout}s: {command}")
        state.command_failures.append(f"{description} (timeout)")
        return 124


def run_sequence(state: AuditState) -> None:
    log_step("Execution Sequence")
    commands = [
        (["pre-commit", "run", "--all-files"], "pre-commit run", True, None, True),
        (["make", "ingest"], "make ingest", False, None, True),
        (["make", "build"], "make build", False, None, True),
        (["make", "score"], "make score", False, None, True),
        (["make", "ml"], "make ml", False, None, True),
        (["make", "reco"], "make reco", False, None, True),
        (["make", "app-smoke"], "make app-smoke", False, {"STREAMLIT_SERVER_HEADLESS": "1"}, True),
        (["make", "format"], "make format", False, None, True),
        (["make", "test"], "make test", False, None, True),
    ]
    for cmd, desc, allow_failure, env, force_allow in commands:
        timeout = 15 if desc == "make app-smoke" else None
        rc = run_command_logged(state, cmd, desc, timeout=timeout, env=env, allow_failure=allow_failure or force_allow)
        if desc == "make app-smoke":
            state.app_smoke_status = "PASS" if rc == 0 else "FAIL"
        if desc == "make test":
            state.test_summary = "PASS" if rc == 0 else f"FAIL (exit {rc})"


def ensure_data_artifacts(state: AuditState) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    log_step("Data Artifact Audit")
    processed_dir = ROOT / "data" / "processed"
    foods_path = processed_dir / "foods_nutrients.parquet"
    nn_path = processed_dir / "nn_index.parquet"
    missing = []
    dataset = None
    nn_index = None
    if not foods_path.exists():
        missing.append("foods_nutrients.parquet")
        run_command_logged(state, ["make", "build"], "make build", allow_failure=False)
        run_command_logged(state, ["make", "score"], "make score", allow_failure=False)
    if not nn_path.exists():
        missing.append("nn_index.parquet")
        run_command_logged(state, ["make", "reco"], "make reco", allow_failure=False)
    if foods_path.exists():
        dataset = pd.read_parquet(foods_path)
    if nn_path.exists():
        nn_index = pd.read_parquet(nn_path)
    passed = not missing
    detail = "All artifacts present" if passed else f"Attempted regeneration for: {', '.join(missing)}"
    state.record("Data artifacts", foods_path.exists() and nn_path.exists(), detail, critical=True)
    return dataset, nn_index


def ensure_perserving_columns(dataset: pd.DataFrame, state: AuditState) -> Tuple[pd.DataFrame, bool]:
    changed = False
    if "serving_gram_weight" not in dataset.columns:
        return dataset, changed
    weights = dataset["serving_gram_weight"].replace({0: np.nan})
    for nutrient in CANONICAL_NUTRIENTS:
        per100 = f"{nutrient}_per100g"
        per_serv = f"{nutrient}_perserving"
        if per100 in dataset.columns and per_serv not in dataset.columns:
            dataset[per_serv] = dataset[per100] * (weights / 100.0)
            append_log(f"Computed missing column: {per_serv}")
            changed = True
    return dataset, changed


def schema_checks(state: AuditState, dataset: pd.DataFrame, nn_index: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log_step("Schema Checks")
    processed_dir = ROOT / "data" / "processed"
    foods_path = processed_dir / "foods_nutrients.parquet"
    nn_path = processed_dir / "nn_index.parquet"

    if dataset is None or dataset.empty:
        state.record("Dataset load", False, "foods_nutrients.parquet missing or empty", critical=True)
        return dataset, nn_index

    dataset, changed = ensure_perserving_columns(dataset, state)
    required_cols = [
        "fdc_id",
        "description",
        "data_type",
        "form",
        "serving_desc",
        "serving_gram_weight",
        "kcal_per_serv",
        "grade",
    ]
    detail_messages = []
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        detail_messages.append(f"Missing columns: {', '.join(missing_cols)}")
    if "nutrition_score" in dataset.columns and "score" not in dataset.columns:
        dataset["score"] = dataset["nutrition_score"]
        changed = True
    if "score" not in dataset.columns:
        detail_messages.append("Missing score column")
    if not dataset["fdc_id"].is_unique:
        before = len(dataset)
        dataset = dataset.drop_duplicates(subset=["fdc_id"], keep="first")
        append_log(f"Deduplicated fdc_id rows: {before - len(dataset)} removed")
        changed = True
    if "score" in dataset.columns:
        score = dataset["score"]
        if score.isna().any():
            dataset["score"] = score.fillna(0.0)
            append_log("Filled NaN scores with 0")
            changed = True
        state.score_min = float(dataset["score"].min())
        state.score_mean = float(dataset["score"].mean())
        state.score_max = float(dataset["score"].max())
        if not ((dataset["score"] >= 0).all() and (dataset["score"] <= 100).all()):
            detail_messages.append("Score outside [0,100]")
    nutrient_missing = []
    for nutrient in CANONICAL_NUTRIENTS:
        per100 = f"{nutrient}_per100g"
        per_serv = f"{nutrient}_perserving"
        if per100 not in dataset.columns or per_serv not in dataset.columns:
            nutrient_missing.append(nutrient)
    if nutrient_missing:
        detail_messages.append(f"Nutrient columns missing for: {', '.join(nutrient_missing)}")
    completeness_cols = [
        f"{nutrient}_per100g" for nutrient in CANONICAL_NUTRIENTS
    ] + [f"{nutrient}_perserving" for nutrient in CANONICAL_NUTRIENTS]
    available_cols = [col for col in completeness_cols if col in dataset.columns]
    if available_cols:
        complete_rows = dataset[available_cols].notna().all(axis=1).mean()
        state.required_nutrient_completeness = float(complete_rows)
    state.dataset_rows = len(dataset)
    if changed:
        dataset.to_parquet(foods_path, index=False)
        append_log("Persisted updates to foods_nutrients.parquet")
        state.log_modification(foods_path)
    state.record("Dataset schema", not detail_messages, "; ".join(detail_messages) if detail_messages else "Schema validated", critical=True)

    if nn_index is None or nn_index.empty:
        state.record("NN index", False, "nn_index.parquet missing or empty", critical=True)
    else:
        required_nn_cols = {"fdc_id", "neighbor_fdc_id", "similarity"}
        if not required_nn_cols.issubset(nn_index.columns):
            state.record("NN index", False, "Missing columns in nn_index.parquet", critical=True)
        else:
            similarity = nn_index["similarity"]
            valid_range = ((similarity > 0) & (similarity <= 1)).all()
            state.record("NN index", bool(valid_range), "Similarities within (0,1]" if valid_range else "Similarity outside expected range", critical=True)
    return dataset, nn_index


def ml_audit(state: AuditState, dataset: pd.DataFrame) -> None:
    log_step("ML Audit")
    required_paths = [
        ROOT / "models" / "coef.json",
        ROOT / "src" / "ml" / "feature_view.py",
        ROOT / "src" / "ml" / "predict.py",
        ROOT / "src" / "service" / "predictor.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    if missing:
        state.record("ML files", False, f"Missing: {', '.join(missing)}", critical=True)
        return
    sys.path.insert(0, str(ROOT))
    try:
        from src.ml import predict as ml_predict

        processed_dir = ROOT / "data" / "processed"
        models_dir = ROOT / "models"
        sample_ids = (
            dataset.sort_values("score", ascending=False)["fdc_id"]
            .dropna()
            .astype(int)
            .unique()
        )
        if len(sample_ids) == 0:
            raise ValueError("No FDC IDs available for ML audit")
        sample_ids = sample_ids[:5]
        scores = []
        probs = []
        for fdc_id in sample_ids:
            result = ml_predict.predict_for_fdc_id(int(fdc_id), processed_dir, models_dir)
            scores.append(result["score"])
            prob_ab = result["probabilities"].get("prob_AB", 0.0)
            probs.append(prob_ab)
        sorted_pairs = sorted(zip(scores, probs))
        monotonic = all(b2 >= b1 - 1e-6 for (_, b1), (_, b2) in zip(sorted_pairs, sorted_pairs[1:]))
        state.record("ML predictions", monotonic, "Monotonic AB probability" if monotonic else "Non-monotonic AB probability", critical=True)
    except Exception as exc:  # noqa: BLE001
        append_log(f"ML audit error: {exc}")
        state.record("ML predictions", False, f"Exception: {exc}", critical=True)


def recommender_audit(state: AuditState, dataset: pd.DataFrame, nn_index: pd.DataFrame) -> None:
    log_step("Recommender Audit")
    if dataset is None or nn_index is None or dataset.empty or nn_index.empty:
        state.record("Recommender", False, "Missing dataset or nn index", critical=True)
        return
    try:
        from src import reco

        low_score = (
            dataset.sort_values("score", ascending=True)["fdc_id"]
            .dropna()
            .astype(int)
            .unique()
        )
        sample_ids = low_score[:5]
        processed_dir = ROOT / "data" / "processed"
        good = True
        for fdc_id in sample_ids:
            swaps = reco.get_substitutions(int(fdc_id), processed_dir=processed_dir, dataset=dataset, index=nn_index)
            if swaps.empty or (swaps["score_gain"].fillna(0) < 10).all():
                good = False
                break
        state.record("Recommender swaps", good, "Valid swaps" if good else "Insufficient score gain", critical=True)

        target = dataset[dataset["grade"].isin(["D", "E"])]
        sample = target.head(2000)
        covered = 0
        for fdc_id in sample["fdc_id"].astype(int):
            swaps = reco.get_substitutions(int(fdc_id), processed_dir=processed_dir, dataset=dataset, index=nn_index)
            if swaps.empty:
                continue
            if (swaps["score_gain"].fillna(0) >= 10).any():
                covered += 1
        coverage = covered / max(len(sample), 1)
        state.reco_coverage = float(coverage)
        state.record("Recommender coverage", coverage >= 0.9, f"Coverage {coverage:.3f}", critical=True)
    except Exception as exc:  # noqa: BLE001
        append_log(f"Recommender audit error: {exc}")
        state.record("Recommender swaps", False, f"Exception: {exc}", critical=True)


def ensure_tests(state: AuditState) -> None:
    log_step("Tests Presence Audit")
    missing = []
    for rel in REQUIRED_TESTS:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                textwrap.dedent(
                    f"""\
from __future__ import annotations

def test_placeholder():
    assert False, "Auto-generated placeholder test for {rel}"
"""
                ),
                encoding="utf-8",
            )
            append_log(f"Created placeholder test: {rel}")
            state.log_modification(path)
    detail = f"Missing tests created: {', '.join(missing)}" if missing else "All tests present"
    state.record("Tests presence", not bool(missing), detail, critical=True)


def docs_audit(state: AuditState, dataset: pd.DataFrame) -> None:
    log_step("Documentation Audit")

    def ensure_schema_doc() -> None:
        path = ROOT / "docs" / "schema.md"
        if dataset is None or dataset.empty:
            return
        columns = dataset.dtypes.astype(str).to_dict()
        lines = ["# foods_nutrients.parquet schema", ""]
        lines.append(f"Total rows: {len(dataset):,}")
        lines.append("")
        lines.append("| Column | dtype |")
        lines.append("| --- | --- |")
        for name, dtype in sorted(columns.items()):
            lines.append(f"| {name} | {dtype} |")
        try:
            from src.utils_nutrients import get_nutrient_mapping

            mapping = get_nutrient_mapping()
            lines.append("")
            lines.append("## Nutrient mapping")
            lines.append("")
            lines.append("| canonical | nutrient_id | nutrient_name | unit |")
            lines.append("| --- | --- | --- | --- |")
            for canonical, sel in mapping.items():
                lines.append(
                    f"| {canonical} | {sel.nutrient_id} | {sel.nutrient_name} | {sel.unit_name} |"
                )
        except Exception as exc:  # noqa: BLE001
            lines.append("")
            lines.append(f"_Nutrient mapping unavailable: {exc}_")
        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        append_log("Updated docs/schema.md")
        state.log_modification(path)

    def ensure_scoring_doc() -> None:
        path = ROOT / "docs" / "scoring.md"
        config = yaml.safe_load((ROOT / "configs" / "scoring.yml").read_text(encoding="utf-8"))
        lines = [
            "# Nutrition Scoring",
            "",
            "## Weights",
            "",
            json.dumps(config.get("weights", {}), indent=2),
            "",
            "## Daily Values",
            "",
        ]
        for nutrient, value in config.get("daily_values", {}).items():
            lines.append(f"- **{nutrient}**: {value}")
        lines.append("")
        lines.append("## Grade Cutpoints")
        lines.append("")
        for grade, cutoff in config.get("grade_cutpoints", {}).items():
            lines.append(f"- {grade}: ≥ {cutoff}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        append_log("Updated docs/scoring.md")
        state.log_modification(path)

    def ensure_model_card() -> None:
        path = ROOT / "docs" / "model_card.md"
        lines = [
            "# Model Card",
            "",
            "## Overview",
            "Rule-based nutrition score calibrated with logistic regression and interpreted with elastic net coefficients.",
            "",
            "## Data",
            "- Source: USDA FoodData Central CSV extracts located in `./DATA`.",
            "- Features: per-100g nutrients for macros, micros, and sugar penalties.",
            "",
            "## Training",
            "- Logistic regression calibration with 5-fold CV on the rule-based score.",
            "- Elastic Net to surface nutrient drivers of grades A/B.",
            "",
            "## Intended Use",
            "- Comparative guidance for exploring healthier substitutions.",
            "",
            "## Limitations",
            "- Does not account for allergens or full ingredient lists.",
            "- Added sugar estimates rely on total sugar when specific values are missing.",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        append_log("Updated docs/model_card.md")
        state.log_modification(path)

    ensure_schema_doc()
    ensure_scoring_doc()
    ensure_model_card()
    state.record("Documentation", True, "Docs refreshed")


def audit_readme(state: AuditState) -> None:
    log_step("README Audit")
    readme_path = ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    appended_sections = []
    for section in REQUIRED_README_SECTIONS:
        if section not in content:
            appended_sections.append(section)
            content += f"\n\n{section}\n\n_TODO: populate section._\n"
    if appended_sections:
        readme_path.write_text(content, encoding="utf-8")
        state.log_modification(readme_path)
        append_log(f"Appended missing README sections: {', '.join(appended_sections)}")
    state.record("README sections", not appended_sections, "All sections present" if not appended_sections else f"Added: {', '.join(appended_sections)}")


def final_checks(state: AuditState) -> None:
    log_step("Final format/test")
    run_command_logged(state, ["make", "format"], "make format", allow_failure=False)
    rc = run_command_logged(state, ["make", "test"], "make test", allow_failure=False)
    state.test_summary = "PASS" if rc == 0 else f"FAIL (exit {rc})"
    if rc != 0:
        state.critical_failures.append("Tests")


def write_report(state: AuditState) -> None:
    log_step("Writing audit report")
    headers = ["Check", "Status", "Detail"]
    rows = []
    for name, passed, detail in state.checks:
        rows.append([name, "PASS" if passed else "FAIL", detail])
    table = format_table(headers, rows)
    metrics_lines = [
        f"- Rows in foods_nutrients.parquet: {state.dataset_rows:,}",
        f"- Required nutrient completeness: {state.required_nutrient_completeness:.2%}",
        f"- Score distribution: min={state.score_min:.1f}, mean={state.score_mean:.1f}, max={state.score_max:.1f}",
        f"- Recommender coverage (D/E): {state.reco_coverage:.2%}",
        f"- Tests: {state.test_summary}",
        f"- App smoke: {state.app_smoke_status}",
    ]
    log_pointer = f"See `{LOG_PATH.relative_to(ROOT)}` for full command output."
    content = "\n".join(
        [
            "# Audit Report",
            "",
            "## Summary",
            "",
            table,
            "",
            "## Metrics",
            "",
            "\n".join(metrics_lines),
            "",
            "## Notes",
            "",
            log_pointer,
        ]
    )
    REPORT_PATH.write_text(content, encoding="utf-8")
    state.log_modification(REPORT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    discover_project_root()
    state = AuditState()
    ensure_structure(state)
    audit_makefile(state)
    ensure_configs(state)
    run_sequence(state)
    dataset, nn_index = ensure_data_artifacts(state)
    if dataset is None:
        dataset = pd.DataFrame()
    if nn_index is None:
        nn_index = pd.DataFrame()
    dataset, nn_index = schema_checks(state, dataset, nn_index)
    if not dataset.empty:
        ml_audit(state, dataset)
        recommender_audit(state, dataset, nn_index)
    ensure_tests(state)
    docs_audit(state, dataset)
    audit_readme(state)
    final_checks(state)
    write_report(state)

    if state.critical_failures:
        append_log(f"Critical failures: {', '.join(state.critical_failures)}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
