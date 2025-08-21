from __future__ import annotations

import platform
import logging
from pathlib import Path

import pandas as pd

from src.cluster_trainer import (
    load_standardized_features,
    load_scaler,
    run_clustering_pipeline,
    save_clustering_artifacts,
)
from src import schema_config as sc
from src.config import BASE_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s] %(message)s")

# Entradas (da etapa 5)
STD_PATH = BASE_PATH / "data_outputs" / "customer_features_standardized.parquet"
SCALER_PATH = BASE_PATH / "data_outputs" / "models" / "standard_scaler.joblib"

# Saídas
OUT_DIR = BASE_PATH / "data_outputs" / "clustering"

# Colunas originais usadas no padrão (as mesmas da etapa 5)
USED_COLS = [
    "frequency_12m",
    "n_unique_routes_out",
    "avg_price_per_ticket",
    "top_route_out_share",
    "top_company_out_share",
]

# Hiperparâmetros
ALGO = "minibatch"        # "kmeans" ou "minibatch"
K_VALUES = range(2, 11)   # {2..10}
PREFER = "silhouette"     # ou "elbow"
DEFAULT_K = 5
RANDOM_STATE = 42

def main():
    logging.info("Python: %s", platform.python_version())
    logging.info("STD_PATH: %s", STD_PATH)
    logging.info("SCALER_PATH: %s", SCALER_PATH)
    logging.info("OUT_DIR: %s", OUT_DIR)

    # 1) carregar dados padronizados e scaler
    df_std = load_standardized_features(STD_PATH, id_col=sc.CONTACT_ID)
    scaler = load_scaler(SCALER_PATH)

    logging.info("df_std shape: %s", df_std.shape)
    logging.info("colunas z (amostra): %s", [c for c in df_std.columns if c.endswith("_z")][:10])

    # 2) rodar pipeline de clustering
    artifacts = run_clustering_pipeline(
        df_std=df_std,
        used_cols=USED_COLS,
        scaler=scaler,
        algo=ALGO,
        k_values=K_VALUES,
        prefer=PREFER,
        default_k=DEFAULT_K,
        random_state=RANDOM_STATE,
    )

    logging.info("k* escolhido: %s", artifacts["k_best"])
    logging.info("assignments shape: %s", artifacts["assignments"].shape)
    logging.info("centroids_z shape: %s", artifacts["centroids_z"].shape)
    logging.info("centroids_unstd shape: %s", artifacts["centroids_unstd"].shape)

    # 3) salvar artefatos
    save_clustering_artifacts(artifacts, OUT_DIR)

    print("[OK] Clustering concluído! Artefatos em:", Path(OUT_DIR).resolve())

if __name__ == "__main__":
    main()