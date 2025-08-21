# src/io_saver.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple
import logging
import shutil

import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------
# Diretórios e utilidades
# -------------------------
def ensure_dir(path: Path) -> Path:
    """
    Garante a existência de um diretório e retorna o próprio Path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_datetime_column(df: pd.DataFrame, datetime_col: str) -> None:
    """
    Valida se datetime_col existe, é datetime64 e não é totalmente nula.
    Lança ValueError com mensagem amigável quando falhar.
    """
    if datetime_col not in df.columns:
        raise ValueError(f"Coluna de data/hora '{datetime_col}' não encontrada no DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        raise ValueError(f"Coluna '{datetime_col}' não é datetime64. Tipo encontrado: {df[datetime_col].dtype}")
    if df[datetime_col].isna().all():
        raise ValueError(f"Coluna '{datetime_col}' está totalmente nula; impossível particionar por ano/mês.")


def _ensure_datetime(
    df: pd.DataFrame,
    datetime_col: str = "purchase_datetime",
    date_fallback: Optional[str] = "date_purchase",
) -> pd.DataFrame:
    """
    Garante a existência de uma coluna datetime para particionamento.
    Se não existir/for inválida, tenta fallback por 'date_fallback'.
    """
    out = df.copy()

    if datetime_col not in out.columns or not pd.api.types.is_datetime64_any_dtype(out[datetime_col]):
        if date_fallback and (date_fallback in out.columns):
            logger.warning(
                "Coluna '%s' ausente/inválida. Usando fallback '%s' como datetime.",
                datetime_col, date_fallback,
            )
            out[datetime_col] = pd.to_datetime(out[date_fallback], errors="coerce")
        else:
            raise ValueError(
                f"Nem '{datetime_col}' (datetime) nem fallback '{date_fallback}' estão disponíveis para particionamento."
            )

    _validate_datetime_column(out, datetime_col)
    return out


# -------------------------
# Escrita básica (segura)
# -------------------------
def save_single_parquet(
    df: pd.DataFrame,
    out_path: Path,
    *,
    compression: str = "snappy",
    engine: str = "pyarrow",
) -> Path:
    """
    Salva um único Parquet com escrita segura (tmp -> rename).
    Retorna o Path final gravado.
    """
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression=compression, engine=engine)
    tmp.replace(out_path)
    logger.info("Parquet salvo em: %s (%d linhas, %d colunas)", out_path, len(df), df.shape[1])
    return out_path


def save_csv(
    df: pd.DataFrame,
    out_path: Path,
    *,
    encoding: str = "utf-8",
    sep: str = ",",
) -> Path:
    """
    Salva CSV com escrita segura (tmp -> rename). Retorna o Path final.
    """
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding=encoding, sep=sep)
    tmp.replace(out_path)
    logger.info("CSV salvo em: %s (%d linhas, %d colunas)", out_path, len(df), df.shape[1])
    return out_path


# -------------------------
# Escrita particionada
# -------------------------
def _add_partitions(
    df: pd.DataFrame,
    datetime_col: str,
    partition_cols: list[str],
) -> pd.DataFrame:
    """
    Adiciona colunas de partição ao DataFrame (ex.: year, month).
    """
    out = df.copy()
    if "year" in partition_cols:
        out["year"] = out[datetime_col].dt.year.astype("Int32")
    if "month" in partition_cols:
        out["month"] = out[datetime_col].dt.month.astype("Int8")
    return out


def _log_partition_counts(df_part: pd.DataFrame, partition_cols: list[str]) -> None:
    """
    Loga a contagem de linhas por partição, útil para auditoria.
    """
    if not partition_cols:
        logger.info("Sem partições. Linhas: %d", len(df_part))
        return

    counts = (
        df_part
        .groupby(partition_cols, dropna=True)
        .size()
        .reset_index(name="rows")
        .sort_values(partition_cols)
    )
    # Log enxuto (primeiras 20 partições)
    preview = counts.head(20).to_dict(orient="records")
    logger.info(
        "Partições (%s) - total=%d; preview(20)=%s",
        ",".join(partition_cols),
        counts["rows"].sum(),
        preview,
    )


def save_partitioned_by_year(
    df: pd.DataFrame,
    root_dir: Path,
    *,
    datetime_col: str = "purchase_datetime",
    date_fallback: Optional[str] = "date_purchase",
    dataset_name: str = "purchases_clean_partitioned",
    use_pyarrow_dataset: bool = True,
    partition_cols: list[str] = None,  # ex.: ["year"] ou ["year","month"]
    overwrite: bool = False,
    compression: str = "snappy",
) -> Path:
    """
    Salva DataFrame particionado em diretório de dataset (estilo Hive):
    root_dir/dataset_name/<col>=<valor>/...

    - Partições configuráveis: ["year"] (default) ou ["year","month"].
    - Se overwrite=True, remove o diretório do dataset antes de salvar.
    - Tenta usar pyarrow.dataset; em falha, aplica fallback manual por loop.

    Retorna o diretório raiz do dataset salvo.
    """
    if partition_cols is None:
        partition_cols = ["year"]

    df_dt = _ensure_datetime(df, datetime_col=datetime_col, date_fallback=date_fallback)
    df_part = _add_partitions(df_dt, datetime_col=datetime_col, partition_cols=partition_cols)
    _log_partition_counts(df_part, partition_cols)

    target_root = ensure_dir(root_dir) / dataset_name
    if overwrite and target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
    ensure_dir(target_root)

    if use_pyarrow_dataset:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df_part, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=str(target_root),
                partition_cols=partition_cols,
                compression=compression,
            )
            logger.info(
                "Dataset particionado escrito via pyarrow em %s por %s.",
                target_root, partition_cols,
            )
            return target_root
        except Exception as e:
            logger.warning(
                "Falha no pyarrow.dataset (%s). Aplicando fallback manual por partição...",
                e,
            )

    # --------
    # Fallback: salvar por grupo de partição manualmente
    # --------
    group_obj = df_part.groupby(partition_cols, dropna=True, sort=True)
    for keys, g in group_obj:
        # keys pode ser escalar (apenas year) ou tupla (year, month)
        if not isinstance(keys, tuple):
            keys = (keys,)
        subdir = target_root
        for col, val in zip(partition_cols, keys):
            subdir = subdir / f"{col}={int(val)}"
        ensure_dir(subdir)

        # Evita regravar sem necessidade no fallback: 1 arquivo por partição
        out_file = subdir / "part-0.parquet"
        tmp = out_file.with_suffix(out_file.suffix + ".tmp")

        # Remove colunas de partição do conteúdo do parquet (ficam no path Hive-style)
        g_to_save = g.drop(columns=[c for c in partition_cols if c in g.columns], errors="ignore")
        g_to_save.to_parquet(tmp, index=False, compression=compression, engine="pyarrow")
        tmp.replace(out_file)

        logger.info("Salvo: %s (%d linhas)", out_file, len(g_to_save))

    logger.info("Dataset particionado (fallback) concluído em %s", target_root)
    return target_root
