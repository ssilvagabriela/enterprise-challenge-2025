from pathlib import Path
import pandas as pd

from src.config import INTERMEDIATE_PATH, OUTPUT_PATH
from src.windowing import build_observation_window, save_window_parquet

def main():
    # parâmetros (você pode carregar via argparse/yaml se preferir)
    BUILD_DATE = "2024-01-01"   # ex.: data de corte para modelos/EDA
    JANELA_MESES = 12           # ex.: últimos 12 meses

    # 1) carregar df_clean do parquet único gerado na etapa 2
    df_clean_path = INTERMEDIATE_PATH / "purchases_clean.parquet"
    df_clean = pd.read_parquet(df_clean_path)

    # 2) construir janela
    df_window, start_date, build_date_ts, _ = build_observation_window(
        df_clean=df_clean,
        build_date=BUILD_DATE,
        janela_meses=JANELA_MESES,
        datetime_col="purchase_datetime",
        date_fallback="date_purchase",
    )

    # 3) salvar janela
    save_window_parquet(
        df_window=df_window,
        out_dir=INTERMEDIATE_PATH,  # pode usar OUTPUT_PATH se preferir
        janela_meses=JANELA_MESES,
        filename_prefix="purchases_clean_window",
        compression="snappy",
    )

    print(
        f"OK - Janela {JANELA_MESES}m: {start_date.date()} .. {build_date_ts.date()} "
        f"| linhas={df_window.shape[0]}"
    )

if __name__ == "__main__":
    main()