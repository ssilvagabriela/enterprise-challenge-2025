import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src import schema_config as sc  # importamos o dicionário externo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# 1) Padronizar tipos
# -----------------------------
def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # datas
    for col in sc.DATE_COLS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
    
    # horários
    for col in sc.TIME_COLS:
        if col in df:
            x = pd.to_datetime(df[col], errors="coerce", format="%H:%M:%S")
            df[col] = x.dt.time
    
    # numéricos
    for col, dtype in sc.NUMERIC_COLS.items():
        if col in df:
            if dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    return df

# -----------------------------
# 2) Normalizar valores artificiais
# -----------------------------
def normalize_artificial_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, bad_values in sc.ARTIFICIAL_NULLS.items():
        if col in df:
            df[col] = df[col].replace(bad_values, np.nan)
    return df

# -----------------------------
# 3) Criar colunas derivadas
# -----------------------------
def create_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if set(sc.DATE_COLS).issubset(df.columns) and set(sc.TIME_COLS).issubset(df.columns):
        dt_str = df[sc.DATE_COLS[0]].dt.strftime("%Y-%m-%d") + " " + df[sc.TIME_COLS[0]].astype(str)
        df["purchase_datetime"] = pd.to_datetime(dt_str, errors="coerce")
    else:
        df["purchase_datetime"] = pd.to_datetime(df[sc.DATE_COLS[0]], errors="coerce")

    df["day_of_week"] = df["purchase_datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("Int8")
    df["hour_of_day"] = df["purchase_datetime"].dt.hour.astype("Int16")
    return df

# -----------------------------
# 4) Regras de negócio
# -----------------------------
def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # route_out
    if sc.ORIGIN_OUT in df and sc.DEST_OUT in df:
        df["route_out"] = df[sc.ORIGIN_OUT].astype(str) + "__" + df[sc.DEST_OUT].astype(str)
    else:
        df["route_out"] = pd.NA

    # has_return
    has_return_fields = (
        (sc.ORIGIN_RET in df and sc.DEST_RET in df)
        and df[sc.ORIGIN_RET].notna() & df[sc.DEST_RET].notna()
    )
    cond_return = has_return_fields if isinstance(has_return_fields, pd.Series) else False
    cond_tickets = (df[sc.TICKETS_COL] >= 2) if sc.TICKETS_COL in df else False

    df["has_return"] = (cond_return & cond_tickets).astype("Int8") if isinstance(cond_return, pd.Series) else 0
    df["is_round_trip"] = df["has_return"]

    # route_return
    if sc.ORIGIN_RET in df and sc.DEST_RET in df:
        df["route_return"] = np.where(
            df["has_return"] == 1,
            df[sc.ORIGIN_RET].astype(str) + "__" + df[sc.DEST_RET].astype(str),
            pd.NA,
        )
    else:
        df["route_return"] = pd.NA

    return df

# -----------------------------------------------------------
# 5) Filtrar registros inválidos (gmv_success ≤ 0 ou tickets ≤ 0)
# -----------------------------------------------------------
def filter_invalid_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    cond_valid_gmv = df[sc.GMV_COL] > 0 if sc.GMV_COL in df else False
    cond_valid_tk  = df[sc.TICKETS_COL] > 0 if sc.TICKETS_COL in df else False
    valid_mask = cond_valid_gmv & cond_valid_tk

    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()

    logging.info(f"Registros válidos: {df_valid.shape[0]} | inválidos: {df_invalid.shape[0]}")
    return df_valid, df_invalid


# -----------------------------------------------------------
# 6) Métricas financeiras (avg_price_per_ticket, log_gmv_success, gmv_bucket)
#    - gmv_bucket via quantis em registros válidos
# -----------------------------------------------------------
def build_financial_metrics(df: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    df = df.copy()

    # preço médio por ticket
    df["avg_price_per_ticket"] = np.where(
        df[sc.TICKETS_COL] > 0,
        df[sc.GMV_COL] / df[sc.TICKETS_COL],
        np.nan,
    )

    # log do gmv
    df["log_gmv_success"] = np.log1p(df[sc.GMV_COL].clip(lower=0))

    # bucket de GMV
    pos = df[sc.GMV_COL] > 0
    try:
        df.loc[pos, "gmv_bucket"] = pd.qcut(
            df.loc[pos, sc.GMV_COL],
            q=n_buckets,
            labels=[f"Q{i}" for i in range(1, n_buckets + 1)],
        )
    except ValueError:
        df.loc[pos, "gmv_bucket"] = pd.cut(
            df.loc[pos, sc.GMV_COL],
            bins=np.unique(np.percentile(df.loc[pos, sc.GMV_COL], np.linspace(0, 100, n_buckets + 1))),
            include_lowest=True,
        ).astype(str)

    df["gmv_bucket"] = df["gmv_bucket"].astype("string")
    return df


# -----------------------------------------------------------
# 7) Remover outliers (avg_price_per_ticket > p99)
# -----------------------------------------------------------
def remove_outliers_by_p99(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    df = df.copy()
    # calcula p99 somente em valores positivos
    mask_pos = df["avg_price_per_ticket"] > 0
    if mask_pos.any():
        p99 = np.nanpercentile(df.loc[mask_pos, "avg_price_per_ticket"], 99)
    else:
        p99 = np.nan

    if np.isnan(p99):
        logging.info("Não foi possível calcular p99 (sem valores positivos). Nenhum outlier removido.")
        return df, df.iloc[0:0].copy(), p99

    outliers_mask = df["avg_price_per_ticket"] > p99
    df_outliers = df[outliers_mask].copy()
    df_kept = df[~outliers_mask].copy()

    logging.info(f"Outliers removidos (avg_price_per_ticket>p99={p99:.2f}): {df_outliers.shape[0]}")
    return df_kept, df_outliers, p99


# -----------------------------------------------------------
# 8) Identificar duplicados por nk_ota_localizer_id
# -----------------------------------------------------------
def identify_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if sc.PRIMARY_KEY not in df.columns:
        logging.warning(f"Coluna chave '{sc.PRIMARY_KEY}' não encontrada. Padrão de duplicidade não aplicado.")
        df["is_duplicated"] = pd.NA
        return df, df.iloc[0:0].copy()

    duplicated_mask = df.duplicated(subset=[sc.PRIMARY_KEY], keep="first")
    df["is_duplicated"] = duplicated_mask.astype("Int8")

    df_dups = df[df["is_duplicated"] == 1].copy()
    df_unique = df[df["is_duplicated"] != 1].copy()

    logging.info(f"Registros duplicados: {df_dups.shape[0]}")
    return df_unique, df_dups


# -----------------------------------------------------------
# 9) Separar dataset de auditoria (gmv_success < 0)
# -----------------------------------------------------------
def split_audit_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if sc.GMV_COL not in df:
        logging.warning(f"Coluna '{sc.GMV_COL}' não encontrada. Auditoria não aplicada.")
        return df, df.iloc[0:0].copy()

    audit_mask = df[sc.GMV_COL] < 0
    df_audit = df[audit_mask].copy()
    df_rest = df[~audit_mask].copy()
    logging.info(f"Registros auditáveis ({sc.GMV_COL}<0): {df_audit.shape[0]}")
    return df_rest, df_audit


# -----------------------------------------------------------
#  Orquestrador da limpeza
#  Retorna um dicionário com artefatos úteis e o df_clean final
# -----------------------------------------------------------
def _check_minimum_columns(df: pd.DataFrame) -> None:
    required = [*sc.DATE_COLS, *sc.TIME_COLS, sc.GMV_COL, sc.TICKETS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas mínimas ausentes: {missing}")

def run_cleaning_pipeline(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    _check_minimum_columns(df_raw)
    # 0) Tipos
    df = standardize_types(df_raw)
    # 1) Normalizações de nulos artificiais
    df = normalize_artificial_nulls(df)
    # 2) Derivadas
    df = create_derived_columns(df)
    # 3) Regras de negócio
    df = apply_business_rules(df)
    # 4) Auditoria de gmv negativo (sem filtrar ainda para não perder o conjunto)
    df_no_neg, df_audit = split_audit_set(df)
    # 5) Filtrar inválidos (gmv<=0 ou tickets<=0)
    df_valid, df_invalid = filter_invalid_records(df_no_neg)
    # 6) Métricas financeiras
    df_fin = build_financial_metrics(df_valid)
    # 7) Duplicados
    df_unique, df_dups = identify_duplicates(df_fin)
    # 8) Outliers de preço médio
    df_clean, df_outliers, p99 = remove_outliers_by_p99(df_unique)

    logging.info(f"Shape final df_clean: {df_clean.shape}")

    return {
        "df_clean": df_clean,
        "df_invalid": df_invalid,
        "df_outliers": df_outliers,
        "df_duplicates": df_dups,
        "df_audit": df_audit,
        "p99_avg_price": pd.DataFrame({"p99_avg_price_per_ticket": [p99]}),
    }