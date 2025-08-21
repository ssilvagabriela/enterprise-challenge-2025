# src/cleaning.py
from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from src import schema_config as sc  # dicionário/constantes do projeto

logger = logging.getLogger(__name__)


# ================================
# 1) Padronizar tipos
# ================================
def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Converte colunas de data e hora para tipos adequados (tolerante a variações).
    - Converte numéricos conforme schema (quando disponível).
    """
    df = df.copy()

    # datas
    for col in getattr(sc, "DATE_COLS", []):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")

    # horários (tolerante a HH:MM e HH:MM:SS)
    for col in getattr(sc, "TIME_COLS", []):
        if col in df:
            # normaliza para HH:MM:SS (preenche/recorta à direita)
            h = df[col].astype(str).str.strip()
            h = h.str.slice(0, 8)  # no máx 8 chars
            # tenta forçar o padrão HH:MM:SS adicionando ":00" quando vier HH:MM
            h = np.where(h.str.len() == 5, h + ":00", h)
            # quando vier vazio -> NaT
            h = pd.to_datetime(h, errors="coerce", format="%H:%M:%S").dt.time
            df[col] = h

    # numéricos (se o projeto mantiver mapa em sc.READ_DTYPES, isso já vem do loader;
    # aqui garantimos coerção extra quando necessário)
    num_map = getattr(sc, "NUMERIC_COLS", {})
    for col, dtype in num_map.items():
        if col in df:
            if dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


# ================================
# 2) Normalizar valores artificiais
# ================================
def normalize_artificial_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui valores artificiais por NA conforme sc.ARTIFICIAL_NULLS
    (mantendo semântica "0" e "1" do dicionário, com tolerância extra).
    """
    df = df.copy()
    for col, bad_values in getattr(sc, "ARTIFICIAL_NULLS", {}).items():
        if col in df:
            df[col] = df[col].replace(bad_values, np.nan)
    return df


# ================================
# 3) Criar colunas derivadas (datetime, dummies temporais)
# ================================
def create_derived_columns(
    df: pd.DataFrame,
    *,
    tz: str = "America/Sao_Paulo",
    datetime_col: str = getattr(sc, "DATETIME_COL", "purchase_datetime"),
) -> pd.DataFrame:
    """
    Combina data+hora (tolerante) para criar 'purchase_datetime' e deriva
    day_of_week/is_weekend/hour_of_day. Localiza no timezone informado.
    """
    df = df.copy()
    date_col = getattr(sc, "DATE_COLS", ["date_purchase"])[0]
    time_cols = getattr(sc, "TIME_COLS", ["time_purchase"])

    if date_col in df:
        if all(col in df for col in time_cols):
            # Reconstrói string YYYY-MM-DD HH:MM:SS
            time_raw = df[time_cols[0]].astype(str).str.strip()
            time_norm = time_raw.str.slice(0, 8)
            time_norm = np.where(time_norm.str.len() == 5, time_norm + ":00", time_norm)
            dt_str = df[date_col].dt.strftime("%Y-%m-%d") + " " + pd.Series(time_norm, index=df.index).astype(str)
            df[datetime_col] = pd.to_datetime(dt_str, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        else:
            df[datetime_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # sem coluna de data; cria vazia
        df[datetime_col] = pd.NaT

    # Localiza no timezone informado, mantendo timestamp "naive" ao final (UTC-agnóstico),
    # mas com campos derivados coerentes ao fuso.
    # Obs: usamos tz_localize -> tz_convert -> remover tz para preservar compatibilidade.
    if df[datetime_col].notna().any():
        try:
            s = df[datetime_col].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            s = s.dt.tz_convert(tz).dt.tz_localize(None)
            df[datetime_col] = s
        except Exception as e:
            logger.warning("Falha ao aplicar timezone '%s' em %s: %s", tz, datetime_col, e)

    # Derivadas temporais
    df["day_of_week"] = df[datetime_col].dt.dayofweek.astype("Int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("Int8")
    df["hour_of_day"] = df[datetime_col].dt.hour.astype("Int16")

    return df


# ================================
# 4) Regras de negócio (rotas, ida/volta)
# ================================
def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    - route_out / route_return
    - has_return / is_round_trip = (origem_volta & destino_volta) e tickets >= 2
    """
    df = df.copy()

    # route_out
    if sc.ORIGIN_OUT in df and sc.DEST_OUT in df:
        df[sc.ROUTE_OUT_COL] = df[sc.ORIGIN_OUT].astype(str) + "__" + df[sc.DEST_OUT].astype(str)
    else:
        df[sc.ROUTE_OUT_COL] = pd.NA

    # has_return com checagens defensivas
    has_ret = False
    if sc.ORIGIN_RET in df and sc.DEST_RET in df:
        has_ret = df[sc.ORIGIN_RET].notna() & df[sc.DEST_RET].notna()

    tickets_ok = False
    if sc.TICKETS_COL in df:
        # garante tipo inteiro nullable para comparação estável
        df[sc.TICKETS_COL] = pd.to_numeric(df[sc.TICKETS_COL], errors="coerce").astype("Int64")
        tickets_ok = df[sc.TICKETS_COL] >= 2

    if isinstance(has_ret, pd.Series) and isinstance(tickets_ok, pd.Series):
        df[sc.HAS_RETURN_COL] = (has_ret & tickets_ok).astype("Int8")
    else:
        df[sc.HAS_RETURN_COL] = pd.Series(0, index=df.index, dtype="Int8")

    df["is_round_trip"] = df[sc.HAS_RETURN_COL].astype("Int8")

    # route_return
    if sc.ORIGIN_RET in df and sc.DEST_RET in df:
        df[sc.ROUTE_RET_COL] = np.where(
            df[sc.HAS_RETURN_COL] == 1,
            df[sc.ORIGIN_RET].astype(str) + "__" + df[sc.DEST_RET].astype(str),
            pd.NA,
        )
    else:
        df[sc.ROUTE_RET_COL] = pd.NA

    return df


# ================================
# 5) Filtrar registros inválidos
# ================================
def filter_invalid_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mantém somente GMV>0 e tickets>0.
    Retorna (df_valid, df_invalid).
    """
    df = df.copy()
    cond_gmv = df[sc.GMV_COL] > 0 if sc.GMV_COL in df else False
    cond_tk = df[sc.TICKETS_COL] > 0 if sc.TICKETS_COL in df else False
    valid_mask = cond_gmv & cond_tk

    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()
    logger.info("Registros válidos: %s | inválidos: %s", df_valid.shape[0], df_invalid.shape[0])
    return df_valid, df_invalid


# ================================
# 6) Métricas financeiras (avg & log)
#    (gmv_bucket será calculado DEPOIS de outliers)
# ================================
def build_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # preço médio por ticket
    df["avg_price_per_ticket"] = np.where(
        (sc.TICKETS_COL in df) & (df[sc.TICKETS_COL] > 0),
        df[sc.GMV_COL] / df[sc.TICKETS_COL],
        np.nan,
    )
    # log do gmv (não-negativo)
    df["log_gmv_success"] = np.log1p(df[sc.GMV_COL].clip(lower=0))
    return df


def build_gmv_bucket(df: pd.DataFrame, n_buckets: int = 5, col: str = sc.GMV_COL) -> pd.DataFrame:
    """
    Cria gmv_bucket por quantis APÓS outliers removidos.
    """
    df = df.copy()
    pos = df[col] > 0
    try:
        df.loc[pos, "gmv_bucket"] = pd.qcut(
            df.loc[pos, col],
            q=n_buckets,
            labels=[f"Q{i}" for i in range(1, n_buckets + 1)],
        )
    except ValueError:
        # fallback quando há limites repetidos
        edges = np.unique(np.percentile(df.loc[pos, col], np.linspace(0, 100, n_buckets + 1)))
        df.loc[pos, "gmv_bucket"] = pd.cut(df.loc[pos, col], bins=edges, include_lowest=True).astype(str)

    df["gmv_bucket"] = df["gmv_bucket"].astype("string")
    return df


# ================================
# 7) Outliers (p99 de avg_price_per_ticket)
# ================================
def remove_outliers_by_p99(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    df = df.copy()
    mask_pos = df["avg_price_per_ticket"] > 0
    if mask_pos.any():
        p99 = np.nanpercentile(df.loc[mask_pos, "avg_price_per_ticket"], 99)
    else:
        p99 = np.nan

    if np.isnan(p99):
        logger.info("Não foi possível calcular p99 (sem valores positivos). Nenhum outlier removido.")
        return df, df.iloc[0:0].copy(), p99

    outliers_mask = df["avg_price_per_ticket"] > p99
    df_outliers = df[outliers_mask].copy()
    df_kept = df[~outliers_mask].copy()
    logger.info("Outliers removidos (avg_price_per_ticket>p99=%.2f): %s", p99, df_outliers.shape[0])
    return df_kept, df_outliers, p99


# ================================
# 8) Duplicados e resolução canônica
# ================================
def identify_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apenas marca duplicados (para auditoria). Resolução acontece em
    `resolve_duplicates_by_rule`.
    """
    df = df.copy()
    key = getattr(sc, "PRIMARY_KEY", "nk_ota_localizer_id")
    if key not in df.columns:
        logger.warning("Coluna chave '%s' não encontrada. 'is_duplicated' marcado como NA.", key)
        df["is_duplicated"] = pd.NA
        return df, df.iloc[0:0].copy()

    duplicated_mask = df.duplicated(subset=[key], keep="first")
    df["is_duplicated"] = duplicated_mask.astype("Int8")
    df_dups = df[df["is_duplicated"] == 1].copy()
    df_unique = df[df["is_duplicated"] != 1].copy()
    logger.info("Registros duplicados: %s", df_dups.shape[0])
    return df_unique, df_dups


def resolve_duplicates_by_rule(
    df: pd.DataFrame,
    *,
    datetime_col: str = getattr(sc, "DATETIME_COL", "purchase_datetime"),
    key_col: str = getattr(sc, "PRIMARY_KEY", "nk_ota_localizer_id"),
    gmv_col: str = sc.GMV_COL,
) -> pd.DataFrame:
    """
    Resolve duplicados por chave de compra seguindo:
      1) maior datetime (mais recente), depois
      2) maior GMV, e por fim
      3) primeira ocorrência.
    """
    if key_col not in df.columns:
        return df

    order_cols = []
    if datetime_col in df.columns:
        order_cols.append(datetime_col)
    if gmv_col in df.columns:
        order_cols.append(gmv_col)

    if not order_cols:
        # sem colunas de ordenação, só drop_duplicates padrão
        return df.drop_duplicates(subset=[key_col], keep="first")

    asc = [False] * len(order_cols)  # desc para "mais recente/maior"
    df_sorted = df.sort_values([key_col, *order_cols], ascending=[True, *asc])
    df_canon = df_sorted.drop_duplicates(subset=[key_col], keep="first").copy()
    return df_canon


# ================================
# 9) Dataset de auditoria (GMV < 0)
# ================================
def split_audit_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if sc.GMV_COL not in df.columns:
        logger.warning("Coluna '%s' não encontrada. Auditoria não aplicada.", sc.GMV_COL)
        return df, df.iloc[0:0].copy()

    audit_mask = df[sc.GMV_COL] < 0
    df_audit = df[audit_mask].copy()
    df_rest = df[~audit_mask].copy()
    logger.info("Registros auditáveis (%s<0): %s", sc.GMV_COL, df_audit.shape[0])
    return df_rest, df_audit


# ================================
# 10) Orquestrador
# ================================
def _check_minimum_columns(df: pd.DataFrame) -> None:
    required = getattr(sc, "REQUIRED_MIN", [*getattr(sc, "DATE_COLS", []), *getattr(sc, "TIME_COLS", []), sc.GMV_COL, sc.TICKETS_COL])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas mínimas ausentes: {missing}. "
            f"Verifique schema_config.REQUIRED_MIN e o dicionário de dados."
        )


def run_cleaning_pipeline(
    df_raw: pd.DataFrame,
    *,
    tz: str = "America/Sao_Paulo",
    n_buckets: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Pipeline completo:
      tipos -> nulos artificiais -> derivadas (com TZ) -> regras de negócio ->
      auditoria (GMV<0) -> filtro(GMV>0 & tickets>0) -> métricas (avg/log) ->
      duplicados (marca) -> resolução canônica -> outliers -> gmv_bucket (APÓS outliers)
    """
    _check_minimum_columns(df_raw)

    # 0) Tipos
    df = standardize_types(df_raw)

    # 1) Nulos artificiais
    df = normalize_artificial_nulls(df)

    # 2) Derivadas (com timezone)
    df = create_derived_columns(df, tz=tz)

    # 3) Regras de negócio
    df = apply_business_rules(df)

    # 4) Auditoria (GMV<0)
    df_no_neg, df_audit = split_audit_set(df)

    # 5) Filtrar inválidos (GMV<=0 ou tickets<=0)
    df_valid, df_invalid = filter_invalid_records(df_no_neg)

    # 6) Métricas financeiras básicas (avg e log)
    df_fin = build_financial_metrics(df_valid)

    # 7) Duplicados (marcar para auditoria) + resolução canônica para seguir o fluxo
    df_marked, df_dups = identify_duplicates(df_fin)
    df_resolved = resolve_duplicates_by_rule(df_marked)

    # 8) Outliers por p99 em avg_price_per_ticket
    df_kept, df_outliers, p99 = remove_outliers_by_p99(df_resolved)

    # 9) Agora sim: gmv_bucket APÓS remover outliers
    df_clean = build_gmv_bucket(df_kept, n_buckets=n_buckets)

    logger.info("Shape final df_clean: %s", df_clean.shape)

    return {
        "df_clean": df_clean,
        "df_invalid": df_invalid,
        "df_outliers": df_outliers,
        "df_duplicates": df_dups,        # mantém lista de dups detectados
        "df_audit": df_audit,            # GMV<0
        "p99_avg_price": pd.DataFrame({"p99_avg_price_per_ticket": [p99]}),
    }
