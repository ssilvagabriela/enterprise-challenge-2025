from __future__ import annotations

import platform
import logging
from pathlib import Path

from src.cluster_labeler import run_labeling_pipeline, save_labeling_outputs
from src.config import BASE_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s] %(message)s")

# Entradas (da etapa 6)
CLUSTER_DIR = BASE_PATH / "data_outputs" / "clustering"
SCALER_PATH = BASE_PATH / "data_outputs" / "models" / "standard_scaler.joblib"

# Mesmas colunas usadas no clustering
USED_COLS = [
    "frequency_12m",
    "n_unique_routes_out",
    "avg_price_per_ticket",
    "top_route_out_share",
    "top_company_out_share",
]

# Limiares para rótulos por z-score
LOW_THR = -0.5
HIGH_THR = 0.5

# Saída
OUT_DIR = CLUSTER_DIR / "labels"

def main():
    logging.info("Python: %s", platform.python_version())
    logging.info("CLUSTER_DIR: %s", CLUSTER_DIR)
    logging.info("SCALER_PATH: %s", SCALER_PATH)
    logging.info("OUT_DIR: %s", OUT_DIR)

    outputs = run_labeling_pipeline(
        clustering_dir=CLUSTER_DIR,
        scaler_path=SCALER_PATH,
        used_cols=USED_COLS,
        low_thr=LOW_THR,
        high_thr=HIGH_THR,
    )

    save_labeling_outputs(outputs, OUT_DIR)

    print("[OK] Rotulagem concluída! Artefatos em:", Path(OUT_DIR).resolve())

if __name__ == "__main__":
    main()