# main.py
from __future__ import annotations
import logging
import sys

import pandas as pd

from src import data_loader, cleaning, io_saver, windowing, feature_builder
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

        # 3) Salvar resultados intermediários
        output_path = config.RUN_OUTPUT_PATH
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                out_file = output_path / f"{name}.parquet"
                io_saver.save_single_parquet(df, out_file)

                # Para df_clean: salvar particionado e seguir com janela + features
                if name == "df_clean":
                    io_saver.save_partitioned_by_year(
                        df,
                        root_dir=output_path,
                        dataset_name="df_clean_partitioned",
                        partition_cols=["year", "month"],
                        overwrite=True,
                    )

                    # 4) Construir e salvar janela de observação (últimos 12 meses)
                    df_window, start_date, build_date, end_date = windowing.build_observation_window(
                        df,
                        build_date=config.DEFAULT_BUILD_DATE,
                        janela_meses=2,
                        post_filter_valid=True,
                        min_rows=1000,
                    )
                    windowing.save_window_parquet(
                        df_window,
                        out_dir=output_path,
                        janela_meses=0.5,
                        filename_prefix="df_clean_window",
                    )

                    # 5) Construir features de cliente e salvar
                    df_feats = feature_builder.build_customer_features(
                        df_window,
                        build_date=config.DEFAULT_BUILD_DATE,
                        model_version=config.MODEL_VERSION,
                    )
                    feature_builder.save_customer_features(
                        df_feats,
                        out_path=config.path_customer_features(output_path),
                    )

        logger.info("Pipeline concluído com sucesso!")

    except Exception as e:
        logger.exception("Falha ao executar pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
