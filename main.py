# main.py
from __future__ import annotations
import logging
import sys

import pandas as pd

from src import data_loader, cleaning
from src import config

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    try:
        # 1) Carregar dados brutos
        logger.info("Iniciando pipeline...")
        df_raw = data_loader.load_purchases("df_t.csv")

        # 2) Rodar pipeline de limpeza
        results = cleaning.run_cleaning_pipeline(df_raw, tz=config.TIMEZONE)

        # 3) Salvar resultados em parquet/csv
        output_path = config.RUN_OUTPUT_PATH
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                file_path = output_path / f"{name}.parquet"
                df.to_parquet(file_path, index=False)
                logger.info("Resultado salvo em %s (%d linhas)", file_path, len(df))

        logger.info("Pipeline concluído com sucesso!")

    except Exception as e:
        logger.exception("Falha ao executar pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
