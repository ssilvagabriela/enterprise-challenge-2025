from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _ensure_datetime(df: pd.DataFrame, datetime_col: str, date_fallback: str | None = None) -> pd.DataFrame:
    """
    Garante a existência de uma coluna datetime utilizável para particionamento.
    Se `datetime_col` não existir ou vier nulo, tenta derivar de `date_fallback`.
    """
    df = df.copy()
    if datetime_col not in df.columns or df[datetime_col].isna().all():
        if date_fallback and date_fallback in df.columns:
            df[datetime_col] = pd.to_datetime(df[date_fallback], errors="coerce")
            logging.warning(f"'{datetime_col}' ausente/nulo. Derivado de '{date_fallback}'.")
        else:
            raise ValueError(
                f"Coluna '{datetime_col}' não encontrada e sem fallback válido. "
                "Ajuste o pipeline ou informe outro campo de data."
            )
    return df

def save_single_parquet(df: pd.DataFrame, out_path: Path, compression: str = "snappy") -> None:
    """
    Salva um único arquivo Parquet com compressão.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression=compression, engine="pyarrow")
    logging.info(f"Parquet salvo em: {out_path}")

def save_partitioned_by_year(
    df: pd.DataFrame,
    root_dir: Path,
    datetime_col: str = "purchase_datetime",
    date_fallback: str | None = "date_purchase",
    dataset_name: str = "purchases_clean_partitioned",
    use_pyarrow_dataset: bool = True,
) -> Path:
    """
    Salva a base particionada por ano.
    - Opção 1 (recomendada): pyarrow.dataset (particionamento nativo).
    - Opção 2 (fallback): loop por ano gravando subpastas.

    Retorna o caminho raiz do dataset particionado.
    """
    df = _ensure_datetime(df, datetime_col, date_fallback=date_fallback).copy()
    df["year"] = df[datetime_col].dt.year.astype("Int32")

    target_root = root_dir / dataset_name
    target_root.mkdir(parents=True, exist_ok=True)

    if use_pyarrow_dataset:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=str(target_root),
                partition_cols=["year"],
                compression="snappy",
            )
            logging.info(f"Dataset particionado por 'year' salvo em: {target_root}")
            return target_root
        except Exception as e:
            logging.warning(f"Falha no particionamento com pyarrow.dataset: {e}. Usando fallback por loop.")

    # Fallback simples: um arquivo por ano em subpastas year=YYYY
    for y, g in df.groupby("year", dropna=True):
        year_folder = target_root / f"year={int(y)}"
        year_folder.mkdir(parents=True, exist_ok=True)
        out_file = year_folder / f"part-0.parquet"
        g.drop(columns=["year"]).to_parquet(out_file, index=False, compression="snappy", engine="pyarrow")
        logging.info(f"Salvo: {out_file} ({g.shape[0]} linhas)")

    return target_root
