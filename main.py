# main.py
from __future__ import annotations
import logging
import sys

import pandas as pd

from src import data_loader, cleaning, io_saver
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

        # 3) Salvar resultados
        output_path = config.RUN_OUTPUT_PATH
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                # Exemplo: salva todos como parquet seguro
                out_file = output_path / f"{name}.parquet"
                io_saver.save_single_parquet(df, out_file)

                # Para o df_clean, também gera versão particionada (ano/mês)
                if name == "df_clean":
                    io_saver.save_partitioned_by_year(
                        df,
                        root_dir=output_path,
                        dataset_name="df_clean_partitioned",
                        partition_cols=["year", "month"],
                        overwrite=True,
                    )

        logger.info("Pipeline concluído com sucesso!")

    except Exception as e:
        logger.exception("Falha ao executar pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
