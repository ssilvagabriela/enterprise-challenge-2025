# main_artifacts.py
from __future__ import annotations

import platform
import logging
from pathlib import Path

from src.config import BASE_PATH
from src.artifacts import run_artifacts_pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Pastas padrão do projeto
OUTPUTS_DIR = BASE_PATH / "data_outputs"
CLUSTERING_DIR = OUTPUTS_DIR / "clustering"
MODELS_DIR = OUTPUTS_DIR / "models"

# Mesmas features usadas no clustering (etapa 5/6)
USED_COLS = [
    "frequency_12m",
    "n_unique_routes_out",
    "avg_price_per_ticket",
    "top_route_out_share",
    "top_company_out_share",
]

# Metadados do modelo (opcionais, aparecem nos artefatos)
MODEL_INFO = {
    "model_version": "v1",
    "algorithm": "KMeans",        # ou "MiniBatchKMeans"
    "k": 5,                       # ajuste conforme etapa 6
}

def main():
    logging.info("Python: %s", platform.python_version())
    logging.info("OUTPUTS_DIR: %s", OUTPUTS_DIR)
    logging.info("CLUSTERING_DIR: %s", CLUSTERING_DIR)
    logging.info("MODELS_DIR: %s", MODELS_DIR)

    paths = run_artifacts_pipeline(
        outputs_dir=OUTPUTS_DIR,
        clustering_dir=CLUSTERING_DIR,
        models_dir=MODELS_DIR,
        used_cols=USED_COLS,
        model_info=MODEL_INFO,
    )

    for k, p in paths.items():
        print(f"[OK] {k}: {Path(p).resolve()}")

if __name__ == "__main__":
    main()
