from __future__ import annotations
import logging
from typing import Tuple, Dict, List
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

from src import schema_config as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


# ============================
# Utilidades de tempo/labels
# ============================
def _ensure_timecols(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas auxiliares: month, quarter, season_br, daypart, is_festive_period."""
    df = df.copy()

    time_col = sc.DATETIME_COL if sc.DATETIME_COL in df.columns else sc.DATE_FALLBACK_COL
    if time_col not in df.columns:
        raise ValueError(f"Não encontrei coluna temporal '{sc.DATETIME_COL}' nem fallback '{sc.DATE_FALLBACK_COL}'.")

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df["month"] = df[time_col].dt.month.astype("Int8")
    df["quarter"] = df[time_col].dt.quarter.astype("Int8")

    # Estações do ano (Brasil, hemisfério sul):
    # Verão: Dez(12), Jan(1), Fev(2); Outono: Mar(3), Abr(4), Mai(5)
    # Inverno: Jun(6), Jul(7), Ago(8); Primavera: Set(9), Out(10), Nov(11)
    season_map = {
        12: "verao", 1: "verao", 2: "verao",
        3: "outono", 4: "outono", 5: "outono",
        6: "inverno", 7: "inverno", 8: "inverno",
        9: "primavera", 10: "primavera", 11: "primavera",
    }
    df["season_br"] = df["month"].map(season_map).astype("string")

    # Daypart (madrugada 0-5, manhã 6-11, tarde 12-17, noite 18-23)
    if sc.HOUR_COL in df.columns:
        hour = df[sc.HOUR_COL].astype("Int16")
    else:
        hour = df[time_col].dt.hour.astype("Int16")

    bins = [-1, 5, 11, 17, 23]
    labels = ["madrugada", "manha", "tarde", "noite"]
    df["daypart"] = pd.cut(hour, bins=bins, labels=labels).astype("string")

    # Festividades (regras simples e replicáveis):
    # - Natal/Reveillon: de 20/12 a 05/01 (qualquer ano)
    # - Carnaval (aproximação): mês de fevereiro
    # - Festas Juninas: mês de junho
    dt = df[time_col]
    month = dt.dt.month
    day = dt.dt.day
    natal_reveillon = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 5))
    carnaval = (month == 2)
    junho = (month == 6)
    df["is_festive_period"] = (natal_reveillon | carnaval | junho).astype("Int8")

    # is_holiday: usar se existir (por ex., se criado antes no pipeline)
    if "is_holiday" not in df.columns:
        df["is_holiday"] = pd.NA  # manter coluna (pode ser preenchida em outra etapa)

    return df


def _share_of_top(value_counts: pd.Series) -> Tuple[str | pd.NA, float]:
    """Retorna (categoria_top, share)."""
    if value_counts.empty:
        return pd.NA, 0.0
    top_item = value_counts.index[0]
    share = float(value_counts.iloc[0]) / float(value_counts.sum())
    return top_item, share


# ============================
# (A) RFM
# ============================
def build_rfm(df: pd.DataFrame, build_date: pd.Timestamp) -> pd.DataFrame:
    """
    RFM no período da janela já recortada:
      - last_purchase: max(date)
      - frequency_12m: nº de pedidos distintos
      - monetary_gmv_12m: soma de GMV
      - recency_days: dias entre build_date e last_purchase
      - avg_ticket_value: monetary / frequency
    """
    df = df.copy()
    # garantias temporais básicas
    time_col = sc.DATETIME_COL if sc.DATETIME_COL in df.columns else sc.DATE_FALLBACK_COL
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    g = df.groupby(sc.CONTACT_ID, as_index=False)

    # nº único de pedidos (usando chave de pedido)
    order_key = sc.PRIMARY_KEY if hasattr(sc, "PRIMARY_KEY") else "nk_ota_localizer_id"
    freq = g[order_key].nunique().rename(columns={order_key: "frequency_12m"})

    # soma do GMV
    mon = g[sc.GMV_COL].sum().rename(columns={sc.GMV_COL: "monetary_gmv_12m"})

    # last_purchase
    last = g[time_col].max().rename(columns={time_col: "last_purchase"})

    rfm = freq.merge(mon, on=sc.CONTACT_ID, how="outer").merge(last, on=sc.CONTACT_ID, how="outer")

    # recency e ticket médio
    rfm["recency_days"] = (pd.to_datetime(build_date) - rfm["last_purchase"]).dt.days
    rfm["avg_ticket_value"] = np.where(rfm["frequency_12m"] > 0, rfm["monetary_gmv_12m"] / rfm["frequency_12m"], np.nan)

    return rfm


# ============================
# (B) Estrutura de viagem
# ============================
def build_trip_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    - pct_round_trip: média(has_return)
    - n_unique_routes_out: nº de route_out distintos
    - top_route_out, top_route_out_share
    """
    df = df.copy()

    def _agg(group: pd.DataFrame) -> pd.Series:
        pct_round = group[sc.HAS_RETURN_COL].astype(float).mean() if sc.HAS_RETURN_COL in group else 0.0
        routes = group[sc.ROUTE_OUT_COL].dropna().astype(str)
        n_routes = routes.nunique()
        top_route, top_share = _share_of_top(routes.value_counts())
        return pd.Series({
            "pct_round_trip": float(pct_round) if not np.isnan(pct_round) else 0.0,
            "n_unique_routes_out": int(n_routes),
            "top_route_out": top_route,
            "top_route_out_share": float(top_share),
        })

    # <<< AQUI: observed=True e include_groups=False >>>
    out = (
        df.groupby(sc.CONTACT_ID, observed=True)
          .apply(_agg, include_groups=False)
          .reset_index()
    )
    return out


# ============================
# (C) Companhia preferida
# ============================
def build_company_pref(df: pd.DataFrame) -> pd.DataFrame:
    """
    - top_company_out, top_company_out_share
    """
    df = df.copy()

    def _agg(group: pd.DataFrame) -> pd.Series:
        comp = group[sc.COMPANY_OUT_COL].dropna().astype(str)
        top_company, top_share = _share_of_top(comp.value_counts())
        return pd.Series({
            "top_company_out": top_company,
            "top_company_out_share": float(top_share),
        })

    # <<< AQUI: observed=True e include_groups=False >>>
    out = (
        df.groupby(sc.CONTACT_ID, observed=True)
          .apply(_agg, include_groups=False)
          .reset_index()
    )
    return out


# ============================
# (D) Sazonalidade
# ============================
def build_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    - pct_weekend_purchases
    - pct_q1..pct_q4
    - pct_verao..pct_primavera
    - pct_madrugada..pct_noite
    - pct_holiday_purchases, pct_festive_period_purchases
    """
    df = _ensure_timecols(df)

    def _proportions(series: pd.Series, categories: List[str]) -> Dict[str, float]:
        vc = series.value_counts(normalize=True)
        return {cat: float(vc.get(cat, 0.0)) for cat in categories}

    # base de proporções
    grp = df.groupby(sc.CONTACT_ID)

    # fim de semana
    if sc.WEEKEND_COL in df:
        weekend_mean = grp[sc.WEEKEND_COL].mean().rename("pct_weekend_purchases").fillna(0.0)
    else:
        weekend_mean = pd.Series(0.0, index=grp.size().index, name="pct_weekend_purchases")

    # quarter
    quarter_props = grp["quarter"].apply(
        lambda s: pd.Series(_proportions(s.astype("Int8").astype(str),
                                         categories=["1", "2", "3", "4"]))
        .rename(index={"1": "pct_q1", "2": "pct_q2", "3": "pct_q3", "4": "pct_q4"})
    ).reset_index()

    # season (Brasil)
    seasons = ["verao", "outono", "inverno", "primavera"]
    season_props = grp["season_br"].apply(
        lambda s: pd.Series(_proportions(s, categories=seasons))
        .rename(index={f: f"pct_{f}" for f in seasons})
    ).reset_index()

    # daypart
    dayparts = ["madrugada", "manha", "tarde", "noite"]
    daypart_props = grp["daypart"].apply(
        lambda s: pd.Series(_proportions(s, categories=dayparts))
        .rename(index={f: f"pct_{f}" for f in dayparts})
    ).reset_index()

    # holiday / festive
    # Nota: is_holiday pode conter NA; tratamos como 0 para média
    if "is_holiday" in df.columns:
        holiday_mean = grp["is_holiday"].apply(
            lambda s: pd.to_numeric(s, errors="coerce")  # garante numérico
                        .fillna(0)                      # preenche como 0
                        .astype("Int8")                 # tipo inteiro pequeno
                        .mean()
        ).rename("pct_holiday_purchases")
    else:
        holiday_mean = pd.Series(0.0, index=grp.size().index, name="pct_holiday_purchases")

    festive_mean = grp["is_festive_period"].mean().rename("pct_festive_period_purchases").fillna(0.0)

    # juntar tudo
    saz = weekend_mean.reset_index()
    for part in [quarter_props, season_props, daypart_props, holiday_mean.reset_index(), festive_mean.reset_index()]:
        saz = saz.merge(part, on=sc.CONTACT_ID, how="left")

    # preencher ausências com 0
    saz = saz.fillna(0.0)

    return saz


# ============================
# (E) Intensidade
# ============================
def build_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    - total_tickets_12m = soma(total_tickets_quantity_success)
    - avg_price_per_ticket = monetary_gmv_12m / total_tickets_12m
    """
    g = df.groupby(sc.CONTACT_ID, as_index=False)

    tickets = g[sc.TICKETS_COL].sum().rename(columns={sc.TICKETS_COL: "total_tickets_12m"})
    gmv = g[sc.GMV_COL].sum().rename(columns={sc.GMV_COL: "monetary_gmv_12m"})

    inten = tickets.merge(gmv, on=sc.CONTACT_ID, how="outer")
    inten["avg_price_per_ticket"] = np.where(
        inten["total_tickets_12m"] > 0,
        inten["monetary_gmv_12m"] / inten["total_tickets_12m"],
        np.nan,
    )
    return inten

def _ensure_unique_per_contact(df: pd.DataFrame, block_name: str) -> pd.DataFrame:
    """Se houver múltiplas linhas por fk_contact em um bloco, agrega por first()."""
    if df.duplicated(subset=[sc.CONTACT_ID]).any():
        logging.warning(f"{block_name}: múltiplas linhas por cliente; agregando por first().")
        df = (
            df.sort_values(sc.CONTACT_ID)
              .groupby(sc.CONTACT_ID, as_index=False, sort=False)
              .first()
        )
    return df

def _ensure_unique_by_contact(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """
    Garante 1 linha por fk_contact. Se houver múltiplas, agrega por first()
    e loga um aviso. Mantém a ordem de colunas recebida.
    """
    if df.duplicated(subset=[sc.CONTACT_ID]).any():
        logger.warning("%s: múltiplas linhas por cliente; agregando por first().", where)
        # agrega por first mantendo somente a primeira ocorrência de cada coluna
        df = (df.groupby(sc.CONTACT_ID, as_index=False, sort=False)
                .first())
    return df


def _safe_fill_zeros(feats: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    """
    Preenche com 0.0 apenas colunas existentes que casem pelos padrões:
    - começa com 'pct_' OU termina com '_12m' OU começa com 'n_unique_'
    """
    if feats.empty:
        return feats

    cols = []
    for c in feats.columns:
        if c == sc.CONTACT_ID:
            continue
        if c.startswith("pct_") or c.endswith("_12m") or c.startswith("n_unique_"):
            cols.append(c)

    if cols:
        # usa .loc e interseção para evitar desalinhamento
        feats.loc[:, cols] = feats.loc[:, cols].fillna(0.0)
    return feats


def _coerce_numeric_safely(feats: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Converte para numérico as colunas que começam com determinado prefixo (ex.: 'pct_'),
    usando conversão coluna a coluna (evita FutureWarning em apply(errors='ignore')).
    """
    target_cols = [c for c in feats.columns if c.startswith(prefix)]
    for c in target_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    return feats

# ============================
# Orquestração
# ============================
def build_customer_features(
    df_cf: pd.DataFrame,
    build_date: str | pd.Timestamp,
    model_version: str = "v1",
) -> pd.DataFrame:
    bd = pd.to_datetime(build_date)

    # --- (A) RFM ---
    rfm = build_rfm(df_cf, bd)
    rfm = _ensure_unique_by_contact(rfm, "rfm")

    # --- (B) Estrutura de viagem ---
    trip = build_trip_structure(df_cf)
    trip = _ensure_unique_by_contact(trip, "trip_structure")

    # --- (C) Companhia preferida ---
    comp = build_company_pref(df_cf)
    comp = _ensure_unique_by_contact(comp, "company_pref")

    # --- (D) Sazonalidade ---
    saz = build_seasonality(df_cf)
    saz = _ensure_unique_by_contact(saz, "seasonality")

    # --- (E) Intensidade ---
    inten = build_intensity(df_cf)
    inten = _ensure_unique_by_contact(inten, "intensity")

    # --- Merge incremental por fk_contact ---
    parts = [rfm, trip, comp, saz, inten]
    feats = parts[0]
    for part in parts[1:]:
        feats = feats.merge(part, on=sc.CONTACT_ID, how="outer", copy=False)

    # Garantir que fk_contact seja único após todos os merges
    feats = _ensure_unique_by_contact(feats, "final_merge")

    # Preencher zeros somente onde faz sentido (proporções/contagens)
    feats = _safe_fill_zeros(feats, patterns=["pct_", "_12m", "n_unique_"])

    # Coagir colunas pct_* para numérico (evita strings como "0.0")
    feats = _coerce_numeric_safely(feats, prefix="pct_")

    # Meta-infos
    feats["build_date"] = bd.normalize()
    feats["model_version"] = model_version

    # Ordena colunas: id, metas, depois features (só estética)
    meta_cols = [sc.CONTACT_ID, "build_date", "model_version"]
    other_cols = [c for c in feats.columns if c not in meta_cols]
    feats = feats[meta_cols + other_cols]

    logger.info("feats shape: %s", feats.shape)
    logger.info("contatos únicos: %d", feats[sc.CONTACT_ID].nunique())

    return feats


def save_customer_features(df_feats: pd.DataFrame, out_path: Path, compression: str = "snappy") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_parquet(out_path, index=False, compression=compression, engine="pyarrow")
    logging.info(f"Features por cliente salvas em: {out_path}")
    return out_path
