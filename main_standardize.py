from __future__ import annotations

import platform
import logging
from pathlib import Path

from src.standardizer import (
    read_customer_features,
    standardize_for_clustering,
    save_standardized_features,
    save_scaler,
)
from src.io_saver import ensure_dir
from src import schema_config as sc
from src.config import BASE_PATH  # já definimos antes

logging.basicConfig(level=logging.INFO, format="%(levelname)s] %(message)s")

# -----------------------------
# Parâmetros do job
# -----------------------------
INPUT_PATH = BASE_PATH / "data_outputs" / "customer_features.parquet"
OUTPUT_STD_PATH = BASE_PATH / "data_outputs" / "customer_features_standardized.parquet"
OUTPUT_SCALER_PATH = BASE_PATH / "data_outputs" / "models" / "standard_scaler.joblib"

# lista default pedida
CLUSTER_VARS = [
    "frequency_12m",
    "n_unique_routes_out",
    "avg_price_per_ticket",
    "top_route_out_share",
    "top_company_out_share",
]

# se quiser forçar colunas de log, troque log_cols=["frequency_12m", ...]
LOG_MODE = "auto"  # "auto" | list[str] | None

def main():
    logging.info(f"Python: {platform.python_version()}")
    logging.info(f"INPUT: {INPUT_PATH}")
    logging.info(f"OUTPUT STD: {OUTPUT_STD_PATH}")

    # 1) ler features por cliente
    df = read_customer_features(INPUT_PATH)
    logging.info(f"df feats shape: {df.shape}")

    # checagem rápida
    miss = [c for c in CLUSTER_VARS if c not in df.columns]
    if miss:
        logging.warning(f"Algumas colunas de cluster não existem no arquivo: {miss}")

    # 2) padronizar (log1p + zscore)
    X_std, scaler, used_cols = standardize_for_clustering(
        df=df,
        cluster_var=CLUSTER_VARS,
        log_cols=LOG_MODE,
        return_scaler=True,
    )
    logging.info(f"X_std shape: {X_std.shape} | usadas: {used_cols}")

    # 3) salvar parquet
    ensure_dir(OUTPUT_STD_PATH.parent)
    save_standardized_features(X_std, OUTPUT_STD_PATH)

    # 4) (opcional) salvar scaler para reuso no pipeline de produção
    if scaler is not None:
        save_scaler(scaler, OUTPUT_SCALER_PATH)

    print("[OK] customer_features_standardized salvo em:", OUTPUT_STD_PATH)

if __name__ == "__main__":
    main()