# src/config.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os

# --- Base do projeto (portável) ---
BASE_PATH = Path(os.getenv("EC_BASE_PATH", Path(__file__).resolve().parents[1]))

# --- Estruturas principais ---
RAW_PATH         = BASE_PATH / "data_raw"
INTERMEDIATE_PATH= BASE_PATH / "data_intermediate"
OUTPUT_PATH      = BASE_PATH / "data_outputs"
MODELS_DIR       = OUTPUT_PATH / "models"
CLUSTERING_DIR   = OUTPUT_PATH / "clustering"
REPORTS_DIR      = OUTPUT_PATH / "reports"
LOGS_DIR         = OUTPUT_PATH / "logs"

for p in [RAW_PATH, INTERMEDIATE_PATH, OUTPUT_PATH, MODELS_DIR, CLUSTERING_DIR, REPORTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- Metadados de execução / padrão do projeto ---
TIMEZONE         = os.getenv("TIMEZONE", "America/Sao_Paulo")
MODEL_VERSION    = os.getenv("MODEL_VERSION", "v1")
DEFAULT_BUILD_DATE = os.getenv("BUILD_DATE", "2024-01-01")
RUN_ID           = os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_OUTPUT_PATH  = OUTPUT_PATH / RUN_ID
RUN_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Helpers (exemplos)
def path_customer_features(run_path: Path | None = None) -> Path:
    base = run_path or OUTPUT_PATH
    return base / "customer_features.parquet"
