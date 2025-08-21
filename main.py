from pathlib import Path
from src.config import INTERMEDIATE_PATH, OUTPUT_PATH
from src.data_loader import load_purchases
from src.cleaning import run_cleaning_pipeline
from src.io_saver import save_single_parquet, save_partitioned_by_year

def main():
    # 0) leitura
    df_raw = load_purchases("df_t.csv")

    # 1) limpeza
    artifacts = run_cleaning_pipeline(df_raw)
    df_clean = artifacts["df_clean"]

    # 2) carga inicial
    # 2.1 arquivo único
    single_parquet = INTERMEDIATE_PATH / "purchases_clean.parquet"
    save_single_parquet(df_clean, single_parquet)

    # 2.2 dataset particionado por ano (usa purchase_datetime gerada no cleaning)
    partition_root = OUTPUT_PATH  # ou INTERMEDIATE_PATH, se preferir
    save_partitioned_by_year(
        df_clean,
        root_dir=partition_root,
        datetime_col="purchase_datetime",
        date_fallback="date_purchase",  # caso alguma linha não tenha horário
        dataset_name="purchases_clean_partitioned",
        use_pyarrow_dataset=True,
    )

    print("Carga inicial concluída.")

if __name__ == "__main__":
    main()
