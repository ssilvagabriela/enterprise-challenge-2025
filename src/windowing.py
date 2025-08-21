# src/windowing.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from src import schema_config as sc
from src.io_saver import save_single_parquet

logger = logging.getLogger(__name__)


# ---------------------------
# Utilidades internas
# ---------------------------
def _coerce_build_date(build_date: str | pd.Timestamp) -> pd.Timestamp:
    """
    Coage a build_date para Timestamp (sem timezone).
    Aceita string 'YYYY-MM-DD' ou Timestamp.
    """
    if isinstance(build_date, pd.Timestamp):
        return build_date.tz_localize(None) if build_date.tzinfo else build_date
    try:
        ts = pd.to_datetime(build_date, errors="raise")
        return ts.tz_localize(None) if ts.tzinfo else ts
    except Exception as e:
        raise ValueError(f"build_date inválida: {build_date!r}. Use 'YYYY-MM-DD'. Erro: {e}")


def _ensure_datetime_col(
    df: pd.DataFrame,
    *,
    datetime_col: str = sc.DATETIME_COL,
    date_fallback: Optional[str] = sc.DATE_FALLBACK_COL,
) -> str:
    """
    Garante a existência de uma coluna temporal (datetime64) para filtro.
    Retorna o nome da coluna a ser usada.
    """
    if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        return datetime_col

    if date_fallback and (date_fallback in df.columns):
        # Converte fallback em datetime
        logger.warning(
            "Coluna '%s' ausente/inválida. Usando '%s' como referência temporal.",
            datetime_col, date_fallback,
        )
        df[date_fallback] = pd.to_datetime(df[date_fallback], errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(df[date_fallback]):
            raise ValueError(f"Fallback '{date_fallback}' não pôde ser convertido para datetime.")
        return date_fallback

    raise ValueError(
        f"Não foi encontrada coluna temporal válida. "
        f"Tente garantir '{datetime_col}' (ou fallback '{date_fallback}')."
    )


# ---------------------------
# Janela de observação
# ---------------------------
def build_observation_window(
    df_clean: pd.DataFrame,
    *,
    build_date: str | pd.Timestamp,
    janela_meses: int = 12,
    datetime_col: str = sc.DATETIME_COL,
    date_fallback: Optional[str] = sc.DATE_FALLBACK_COL,
    post_filter_valid: bool = False,
    min_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Cria uma janela fechada de observação [start_date .. end_of_build_day].
    - 'start_date' = build_date - janela_meses (via DateOffset).
    - 'end_of_build_day' = build_date normalizada para o fim do dia (23:59:59.999999).
    - Seleciona a coluna temporal a partir do schema (preferência: sc.DATETIME_COL).

    Parâmetros:
      df_clean          : DataFrame já tratado (saído do cleaning).
      build_date        : Data limite inclusiva (str 'YYYY-MM-DD' ou Timestamp).
      janela_meses      : Tamanho da janela (meses) para trás.
      datetime_col      : Nome da coluna datetime principal (default: sc.DATETIME_COL).
      date_fallback     : Fallback de data (default: sc.DATE_FALLBACK_COL).
      post_filter_valid : Se True, aplica um filtro adicional de GMV>0 & tickets>0 (segurança).
      min_rows          : Se informado, loga warning quando a janela tiver < min_rows.

    Retorna:
      (df_window, start_date, build_date_norm, end_of_build_day)
    """
    if janela_meses <= 0:
        raise ValueError("janela_meses deve ser > 0.")

    # 1) Normaliza datas
    bd = _coerce_build_date(build_date)
    start_date = bd - pd.DateOffset(months=janela_meses)

    # Fim do dia para incluir compras do dia de build_date
    end_of_build_day = (bd.normalize() + pd.Timedelta(days=1)) - pd.Timedelta(microseconds=1)

    # 2) Seleciona coluna temporal coerente
    time_col = _ensure_datetime_col(df_clean, datetime_col=datetime_col, date_fallback=date_fallback)

    # 3) Filtro da janela [start_date .. end_of_build_day]
    mask = (df_clean[time_col] >= start_date) & (df_clean[time_col] <= end_of_build_day)
    df_window = df_clean.loc[mask].copy()

    # 4) (Opcional) filtro de segurança caso df_clean não tenha passado pelo pipeline completo
    if post_filter_valid:
        if sc.GMV_COL in df_window.columns and sc.TICKETS_COL in df_window.columns:
            df_window = df_window[(df_window[sc.GMV_COL] > 0) & (df_window[sc.TICKETS_COL] > 0)].copy()
        else:
            logger.warning(
                "post_filter_valid=True, mas colunas '%s'/'%s' não estão presentes. Filtro ignorado.",
                sc.GMV_COL, sc.TICKETS_COL,
            )

    # 5) Sanidade e logs
    if min_rows is not None and len(df_window) < min_rows:
        logger.warning(
            "Janela com poucas linhas: %d < %d. Datas: [%s .. %s] usando coluna '%s'.",
            len(df_window), min_rows, start_date.date(), end_of_build_day.date(), time_col,
        )

    logger.info(
        "Janela construída: [%s .. %s] | linhas=%d | col_tempo='%s' | meses=%d",
        start_date.date(), end_of_build_day.date(), df_window.shape[0], time_col, janela_meses
    )

    return df_window, start_date, bd.normalize(), end_of_build_day


# ---------------------------
# Salvamento da janela
# ---------------------------
def save_window_parquet(
    df_window: pd.DataFrame,
    out_dir: Path,
    *,
    janela_meses: int,
    filename_prefix: str = "purchases_clean_window",
    compression: str = "snappy",
) -> Path:
    """
    Salva o DataFrame de janela em Parquet usando escrita segura (tmp->rename).
    Retorna o Path final escrito.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{janela_meses}m.parquet"
    save_single_parquet(df_window, out_path, compression=compression)
    logger.info("Janela salva em %s (%d linhas)", out_path, len(df_window))
    return out_path