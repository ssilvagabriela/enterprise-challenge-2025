# src/feature_builder.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from src import schema_config as sc
from src.io_saver import save_single_parquet

logger = logging.getLogger(__name__)


# =========================================================
# Utilidades
# =========================================================
def _coerce_build_date(build_date: str | pd.Timestamp) -> pd.Timestamp:
    """Coage build_date para Timestamp naive (sem tz)."""
    if isinstance(build_date, pd.Timestamp):
        return build_date.tz_localize(None) if build_date.tzinfo else build_date
    ts = pd.to_datetime(build_date, errors="raise")
    return ts.tz_localize(None) if ts.tzinfo else ts


def _group_apply_with_fallback(grouper, func):
    """Compatibilidade com pandas antigos (sem include_groups)."""
    try:
        return grouper.apply(func, include_groups=False)
    except TypeError:
        return grouper.apply(func)


def _top_and_share(s: pd.Series) -> Tuple[str | pd.NA, float]:
    """Retorna (valor_top, share_top) considerando valores não nulos."""
    if s.dropna().empty:
        return pd.NA, 0.0
    vc = s.value_counts(dropna=True)
    top_val = vc.index[0]
    share = float(vc.iloc[0] / vc.sum()) if vc.sum() > 0 else 0.0
    return str(top_val), share


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return (num / den).fillna(0.0)


# =========================================================
# Pré-processo temporal para sazonalidade
# =========================================================
def _ensure_timecols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante colunas auxiliares para sazonalidade:
      - month, quarter, season_br (verão/outono/inverno/primavera),
      - daypart (madrugada, manhã, tarde, noite),
      - is_weekend (se ainda não existir),
      - is_holiday (janelas brasileiras simples: Natal/Reveillon, Carnaval, Junho).
    """
    out = df.copy()
    time_col = sc.DATETIME_COL if sc.DATETIME_COL in out else sc.DATE_FALLBACK_COL
    if time_col not in out:
        raise KeyError(f"Coluna temporal '{sc.DATETIME_COL}' ou fallback '{sc.DATE_FALLBACK_COL}' ausente.")

    out["month"] = out[time_col].dt.month.astype("Int8")
    out["quarter"] = out[time_col].dt.quarter.astype("Int8")

    # Estações (hemisfério sul): Dez-Fev (verão), Mar-Mai (outono), Jun-Ago (inverno), Set-Nov (primavera)
    month = out["month"]
    season = pd.Series(index=out.index, dtype="string")
    season[(month >= 12) | (month <= 2)] = "verao"
    season[(month >= 3) & (month <= 5)] = "outono"
    season[(month >= 6) & (month <= 8)] = "inverno"
    season[(month >= 9) & (month <= 11)] = "primavera"
    out["season_br"] = season.astype("string")

    # Daypart (madrugada 0-5, manhã 6-11, tarde 12-17, noite 18-23)
    hour = out[time_col].dt.hour
    bins = [-1, 5, 11, 17, 23]
    labels = ["madrugada", "manha", "tarde", "noite"]
    out["daypart"] = pd.cut(hour, bins=bins, labels=labels).astype("string")

    # is_weekend se não existe
    if "is_weekend" not in out.columns:
        out["is_weekend"] = out[time_col].dt.dayofweek.isin([5, 6]).astype("Int8")

    # is_holiday (heurística simples e reproduzível)
    # - Natal/Reveillon: 20/12..05/01
    # - Carnaval: janela móvel simplificada (fev até início mar: 01/02..15/03)
    # - Junho (festas juninas): 01/06..30/06
    d = out[time_col]
    mmdd = d.dt.strftime("%m-%d")
    is_xmas_newyear = (mmdd >= "12-20") | (mmdd <= "01-05")
    is_carnival = (mmdd >= "02-01") & (mmdd <= "03-15")
    is_june = (mmdd >= "06-01") & (mmdd <= "06-30")
    out["is_holiday"] = (is_xmas_newyear | is_carnival | is_june).astype("Int8")

    return out


# =========================================================
# Bloco A — RFM
# =========================================================
def build_rfm(df: pd.DataFrame, build_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Retorna colunas:
      - last_purchase (Timestamp),
      - frequency_12m (nunique de pedidos),
      - monetary_gmv_12m (soma),
      - recency_days (build_date - last_purchase),
      - avg_ticket_value (monetary / frequency).
    """
    bd = _coerce_build_date(build_date)
    time_col = sc.DATETIME_COL if sc.DATETIME_COL in df else sc.DATE_FALLBACK_COL
    key_col = getattr(sc, "PRIMARY_KEY", "nk_ota_localizer_id")

    g = df.groupby(sc.CONTACT_ID, observed=True)
    rfm = g.agg(
        last_purchase=(time_col, "max"),
        frequency_12m=(key_col, "nunique"),
        monetary_gmv_12m=(sc.GMV_COL, "sum"),
    ).reset_index()

    rfm["recency_days"] = (bd - rfm["last_purchase"]).dt.days.astype("Int32")
    rfm["avg_ticket_value"] = np.where(
        rfm["frequency_12m"] > 0,
        rfm["monetary_gmv_12m"] / rfm["frequency_12m"],
        np.nan,
    )
    return rfm


# =========================================================
# Bloco B — Estrutura de viagem (ida/volta e rotas)
# =========================================================
def build_trip_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna por cliente:
      - pct_round_trip,
      - n_unique_routes_out,
      - top_route_out,
      - top_route_out_share.
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        pct_round = g[sc.HAS_RETURN_COL].fillna(0).astype("Int8").mean() if sc.HAS_RETURN_COL in g else 0.0
        n_routes = g[sc.ROUTE_OUT_COL].nunique(dropna=True) if sc.ROUTE_OUT_COL in g else 0
        top, share = _top_and_share(g[sc.ROUTE_OUT_COL]) if sc.ROUTE_OUT_COL in g else (pd.NA, 0.0)
        return pd.Series(
            {
                "pct_round_trip": float(pct_round),
                "n_unique_routes_out": int(n_routes),
                "top_route_out": top,
                "top_route_out_share": float(share),
            }
        )

    out = _group_apply_with_fallback(df.groupby(sc.CONTACT_ID, observed=True), _agg).reset_index()
    # Tipos estáveis
    out["pct_round_trip"] = out["pct_round_trip"].astype("float64")
    out["n_unique_routes_out"] = out["n_unique_routes_out"].astype("Int32")
    out["top_route_out_share"] = out["top_route_out_share"].astype("float64")
    out["top_route_out"] = out["top_route_out"].astype("string")
    return out


# =========================================================
# Bloco C — Companhia preferida (ida)
# =========================================================
def build_company_pref(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna por cliente:
      - top_company_out,
      - top_company_out_share.
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        if sc.COMPANY_OUT_COL in g:
            top, share = _top_and_share(g[sc.COMPANY_OUT_COL])
        else:
            top, share = (pd.NA, 0.0)
        return pd.Series(
            {"top_company_out": top, "top_company_out_share": float(share)}
        )

    out = _group_apply_with_fallback(df.groupby(sc.CONTACT_ID, observed=True), _agg).reset_index()
    out["top_company_out"] = out["top_company_out"].astype("string")
    out["top_company_out_share"] = out["top_company_out_share"].astype("float64")
    return out


# =========================================================
# Bloco D — Sazonalidade (percentuais)
# =========================================================
def build_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna por cliente os percentuais (0..1):
      - pct_weekend,
      - pct_q1, pct_q2, pct_q3, pct_q4,
      - pct_verao, pct_outono, pct_inverno, pct_primavera,
      - pct_madrugada, pct_manha, pct_tarde, pct_noite,
      - pct_holiday.
    Garante presença de todas as colunas, com 0.0 quando categoria não ocorre.
    """
    x = _ensure_timecols(df)

    # Base de contagem por cliente
    g = x.groupby(sc.CONTACT_ID, observed=True)
    total = g.size().rename("n").reset_index()

    # weekend
    weekend = x.groupby(sc.CONTACT_ID, observed=True)["is_weekend"].mean().rename("pct_weekend").reset_index()

    # quarter
    q_share = (
        x.groupby([sc.CONTACT_ID, "quarter"], observed=True)
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .rename("share")
        .reset_index()
    )
    q_pivot = q_share.pivot(index=sc.CONTACT_ID, columns="quarter", values="share").fillna(0.0)
    q_pivot = q_pivot.rename(columns={1: "pct_q1", 2: "pct_q2", 3: "pct_q3", 4: "pct_q4"}).reset_index()

    # season_br
    s_share = (
        x.groupby([sc.CONTACT_ID, "season_br"], observed=True)
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .rename("share")
        .reset_index()
    )
    s_pivot = s_share.pivot(index=sc.CONTACT_ID, columns="season_br", values="share").fillna(0.0)
    s_pivot = (
        s_pivot.rename(
            columns={
                "verao": "pct_verao",
                "outono": "pct_outono",
                "inverno": "pct_inverno",
                "primavera": "pct_primavera",
            }
        )
        .reindex(columns=["pct_verao", "pct_outono", "pct_inverno", "pct_primavera"], fill_value=0.0)
        .reset_index()
    )

    # daypart
    d_share = (
        x.groupby([sc.CONTACT_ID, "daypart"], observed=True)
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .rename("share")
        .reset_index()
    )
    d_pivot = d_share.pivot(index=sc.CONTACT_ID, columns="daypart", values="share").fillna(0.0)
    d_pivot = (
        d_pivot.rename(
            columns={
                "madrugada": "pct_madrugada",
                "manha": "pct_manha",
                "tarde": "pct_tarde",
                "noite": "pct_noite",
            }
        )
        .reindex(columns=["pct_madrugada", "pct_manha", "pct_tarde", "pct_noite"], fill_value=0.0)
        .reset_index()
    )

    # holiday
    if "is_holiday" in x.columns:
        hol = (
            x.groupby(sc.CONTACT_ID, observed=True)["is_holiday"]
            .mean()
            .rename("pct_holiday")
            .reset_index()
        )
    else:
        hol = total[[sc.CONTACT_ID]].copy()
        hol["pct_holiday"] = 0.0

    # Merge final
    out = (
        total[[sc.CONTACT_ID]]
        .merge(weekend, on=sc.CONTACT_ID, how="left")
        .merge(q_pivot, on=sc.CONTACT_ID, how="left")
        .merge(s_pivot, on=sc.CONTACT_ID, how="left")
        .merge(d_pivot, on=sc.CONTACT_ID, how="left")
        .merge(hol, on=sc.CONTACT_ID, how="left")
        .fillna(0.0)
    )

    # Tipos estáveis
    pct_cols = [c for c in out.columns if c.startswith("pct_")]
    out[pct_cols] = out[pct_cols].astype("float64")
    return out


# =========================================================
# Bloco E — Intensidade
# =========================================================
def build_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna por cliente:
      - total_tickets_12m,
      - avg_price_per_ticket (média no período).
    """
    g = df.groupby(sc.CONTACT_ID, observed=True)
    inten = g.agg(
        total_tickets_12m=(sc.TICKETS_COL, "sum"),
        avg_price_per_ticket=("avg_price_per_ticket", "mean" if "avg_price_per_ticket" in df else (sc.GMV_COL, "mean")),
    ).reset_index()

    # Tipagem consistente
    inten["total_tickets_12m"] = pd.to_numeric(inten["total_tickets_12m"], errors="coerce").fillna(0).astype("Int64")
    inten["avg_price_per_ticket"] = pd.to_numeric(inten["avg_price_per_ticket"], errors="coerce").astype("float64")
    return inten


# =========================================================
# Orquestrador de features
# =========================================================
def _postprocess_and_fill(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche ausências/NaN de forma controlada e garante tipos consistentes.
    - Campos de proporção/percentual: float64
    - Contagens/somas: Int64 quando apropriado
    """
    out = df_feat.copy()

    # Preencher ausências em colunas específicas
    zero_fill_cols = [c for c in out.columns if c.startswith("pct_")] + [
        "n_unique_routes_out",
        "frequency_12m",
        "total_tickets_12m",
        "top_route_out_share",
        "top_company_out_share",
    ]
    for c in zero_fill_cols:
        if c in out:
            if c.startswith("pct_") or c.endswith("_share"):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float64")
            elif c in {"n_unique_routes_out", "frequency_12m", "total_tickets_12m"}:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64")

    # Coerções finais
    float_cols = ["monetary_gmv_12m", "avg_ticket_value", "avg_price_per_ticket", "pct_round_trip"]
    for c in float_cols:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    # Strings amigáveis
    for c in ["top_route_out", "top_company_out"]:
        if c in out:
            out[c] = out[c].astype("string")

    return out


def build_customer_features(
    df_window: pd.DataFrame,
    *,
    build_date: str | pd.Timestamp,
    model_version: str = "v1",
) -> pd.DataFrame:
    """
    Constrói todas as features por cliente a partir do df_window (janela já aplicada).
    Retorna um único DataFrame (1 linha por fk_contact).
    """
    if sc.CONTACT_ID not in df_window.columns:
        raise KeyError(f"Coluna de cliente '{sc.CONTACT_ID}' ausente no df_window.")

    # Blocos
    rfm = build_rfm(df_window, build_date=build_date)
    trip = build_trip_structure(df_window)
    comp = build_company_pref(df_window)
    seas = build_seasonality(df_window)
    inten = build_intensity(df_window)

    # Merge final
    feats = (
        rfm.merge(trip, on=sc.CONTACT_ID, how="left")
           .merge(comp, on=sc.CONTACT_ID, how="left")
           .merge(seas, on=sc.CONTACT_ID, how="left")
           .merge(inten, on=sc.CONTACT_ID, how="left")
    )

    # Metadados do modelo
    bd = _coerce_build_date(build_date)
    feats["build_date"] = bd.normalize()
    feats["model_version"] = str(model_version)

    # Pós-processo (tipos e fills)
    feats = _postprocess_and_fill(feats)

    # Garantir 1 linha por cliente
    feats = feats.drop_duplicates(subset=[sc.CONTACT_ID], keep="first").reset_index(drop=True)
    logger.info("Customer features construídas: %d clientes, %d colunas", feats.shape[0], feats.shape[1])
    return feats


# =========================================================
# Salvamento
# =========================================================
def save_customer_features(
    df_feats: pd.DataFrame,
    out_path: Path,
    *,
    compression: str = "snappy",
) -> Path:
    """Salva features em Parquet com escrita segura."""
    return save_single_parquet(df_feats, out_path, compression=compression)