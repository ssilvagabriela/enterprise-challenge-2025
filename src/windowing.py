from __future__ import annotations
from pathlib import Path
from typing import Tuple
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _coerce_build_date(build_date: str | pd.Timestamp) -> pd.Timestamp:
    """Aceita 'YYYY-MM-DD' ou Timestamp e normaliza para Timestamp sem tz."""
    if isinstance(build_date, pd.Timestamp):
        return build_date.tz_localize(None) if build_date.tzinfo is not None else build_date
    # string
    ts = pd.to_datetime(build_date, errors="raise", format="%Y-%m-%d")
    return ts


def _ensure_datetime_col(df: pd.DataFrame,
                         datetime_col: str = "purchase_datetime",
                         date_fallback: str | None = "date_purchase") -> str:
    """
    Garante que existe uma coluna datetime válida para filtrar a janela.
    Retorna o nome da coluna a ser usada.
    """
    if datetime_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            # tenta converter (caso alguém tenha salvo como string)
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        if df[datetime_col].notna().any():
            return datetime_col

    if date_fallback and date_fallback in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_fallback]):
            df[date_fallback] = pd.to_datetime(df[date_fallback], errors="coerce")
        if df[date_fallback].notna().any():
            logging.warning(f"Usando fallback temporal '{date_fallback}' para janelas.")
            return date_fallback

    raise ValueError(
        f"Não há coluna temporal utilizável. Verifique '{datetime_col}' ou o fallback '{date_fallback}'."
    )


def build_observation_window(
    df_clean: pd.DataFrame,
    build_date: str | pd.Timestamp,
    janela_meses: int,
    datetime_col: str = "purchase_datetime",
    date_fallback: str | None = "date_purchase",
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Filtra df_clean no intervalo fechado [start_date, build_date].

    Retorna: (df_window, start_date, build_date_ts, end_date_ts)
        - end_date_ts == build_date_ts (apenas para clareza e logs)
    """
    if janela_meses <= 0:
        raise ValueError("janela_meses deve ser > 0.")

    build_date_ts = _coerce_build_date(build_date)
    # pandas.DateOffset lida corretamente com meses variáveis
    start_date = build_date_ts - pd.DateOffset(months=janela_meses)

    # qual coluna temporal usar
    time_col = _ensure_datetime_col(df_clean, datetime_col=datetime_col, date_fallback=date_fallback)

    # filtro fechado: >= start_date e <= build_date
    mask = (df_clean[time_col] >= start_date) & (df_clean[time_col] <= build_date_ts)
    df_window = df_clean.loc[mask].copy()

    logging.info(
        f"Janela de observação: [{start_date.date()} .. {build_date_ts.date()}] "
        f"| linhas={df_window.shape[0]} | col_tempo='{time_col}' | meses={janela_meses}"
    )

    return df_window, start_date, build_date_ts, build_date_ts


def save_window_parquet(
    df_window: pd.DataFrame,
    out_dir: Path,
    janela_meses: int,
    filename_prefix: str = "purchases_clean_window",
    compression: str = "snappy",
) -> Path:
    """
    Salva o df_window com o padrão: purchases_clean_window_{janela_meses}m.parquet
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{janela_meses}m.parquet"
    df_window.to_parquet(out_path, index=False, compression=compression, engine="pyarrow")
    logging.info(f"Janela salva em: {out_path}")
    return out_path
