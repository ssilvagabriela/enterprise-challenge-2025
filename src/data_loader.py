# src/data_loader.py
from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd

from src.config import RAW_PATH
from src import schema_config as sc  # usar READ_DTYPES, PARSE_DATES se definidos

logger = logging.getLogger(__name__)

def load_purchases(
    filename: str = "df_t.csv",
    *,
    path: Path | None = None,
    sep: str = ",",
    encoding: str = "utf-8",
    low_memory: bool = False,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
):
    """
    Lê o CSV de compras com controle de dtypes e datas.
    - Se 'path' não for informado, procura em RAW_PATH/filename.
    - Suporta amostragem (nrows) e leitura em chunks (chunksize).
    """
    file_path = (path / filename) if path else (RAW_PATH / filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logger.info("Lendo dados de %s ...", file_path)

    read_kwargs = dict(
        sep=sep,
        encoding=encoding,
        low_memory=low_memory,
        dtype=getattr(sc, "READ_DTYPES", None),     # opcional, se definido no schema
        parse_dates=getattr(sc, "PARSE_DATES", None),  # ex.: ["date_purchase"]
        usecols=usecols,
        nrows=nrows,
    )

    if chunksize:
        # retorna um gerador de chunks
        logger.info("Leitura em chunks: chunksize=%s", chunksize)
        return pd.read_csv(file_path, chunksize=chunksize, **read_kwargs)

    df = pd.read_csv(file_path, **read_kwargs)
    logger.info("Dados carregados: %d linhas, %d colunas.", df.shape[0], df.shape[1])
    return df
