from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.io_saver import ensure_dir, save_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Carregamento de artefatos
# -----------------------------
def load_artifacts(
    clustering_dir: Path | str,
    scaler_path: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Lê:
      - assignments.parquet
      - centroids_z.parquet
      - centroids_unstd.parquet
      - scaler (joblib)
    """
    base = Path(clustering_dir)
    assign = pd.read_parquet(base / "cluster_assignments.parquet")
    cz = pd.read_parquet(base / "cluster_centroids_z.parquet")
    cu = pd.read_parquet(base / "cluster_centroids_unstd.parquet")
    scaler: StandardScaler = joblib.load(scaler_path)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Objeto carregado não é StandardScaler.")
    return assign, cz, cu, scaler


# -----------------------------
# Médias globais (espaço original)
# -----------------------------
def global_means_from_scaler(
    scaler: StandardScaler,
    used_cols: List[str],
) -> pd.Series:
    """
    Usa o mean_ do StandardScaler como média global por variável no espaço original.
    """
    means = pd.Series(scaler.mean_, index=used_cols, name="global_mean")
    return means


# -----------------------------
# Rotulagem por z-score
# -----------------------------
def label_from_z(z: float, low_thr: float = -0.5, high_thr: float = 0.5) -> str:
    """
    Converte um z-score em rótulo categórico.
    Padrão (didático):
      z < -0.5  -> "Baixo"
      -0.5..0.5 -> "Médio"
      > 0.5     -> "Alto"
    """
    if np.isnan(z):
        return "Médio"  # neutro para ausências
    if z < low_thr:
        return "Baixo"
    if z > high_thr:
        return "Alto"
    return "Médio"


def build_centroid_labels(
    centroids_z: pd.DataFrame,
    used_cols: List[str],
    low_thr: float = -0.5,
    high_thr: float = 0.5,
) -> pd.DataFrame:
    """
    Gera rótulos categóricos por variável para cada cluster, a partir dos centróides em z-score.
    Retorna um DataFrame wide: uma linha por cluster_id, colunas "var_label" para cada variável.
    Ex.: "frequency_12m_label" ∈ {"Baixo","Médio","Alto"}.
    """
    needed = ["cluster_id"] + [f"{c}_z" for c in used_cols]
    missing = [c for c in needed if c not in centroids_z.columns]
    if missing:
        raise ValueError(f"Colunas não encontradas em centroids_z: {missing}")

    lab = centroids_z[["cluster_id"]].copy()
    for c in used_cols:
        zcol = f"{c}_z"
        lab[f"{c}_label"] = centroids_z[zcol].apply(lambda v: label_from_z(float(v), low_thr, high_thr))
    return lab


# -----------------------------
# Nomes legíveis de clusters (opcional)
# -----------------------------
def summarize_label(label: str, var_alias: str) -> str:
    """
    Converte um par (variável, rótulo) em um fragmento legível curto.
    Ex.: ("frequency_12m","Alto") -> "Alta frequência"
    """
    base = var_alias
    if label == "Baixo":
        return f"{base} baixa"
    if label == "Alto":
        # ajusta concordância de algumas palavras comuns
        if base.endswith("frequência"):
            return "Alta frequência"
        if base.endswith("preço"):
            return "Preço alto"
        if base.endswith("rotas"):
            return "Diversificação alta"
        if base.endswith("share rota"):
            return "Rota favorita forte"
        if base.endswith("share viação"):
            return "Viação favorita forte"
        return f"{base} alta"
    return f"{base} média"


def default_aliases(used_cols: List[str]) -> Dict[str, str]:
    """
    Dicionário de aliases curtos para mostrar em nomes legíveis.
    Customize conforme necessidade do negócio.
    """
    aliases = {
        "frequency_12m": "frequência",
        "n_unique_routes_out": "diversificação de rotas",
        "avg_price_per_ticket": "preço",
        "top_route_out_share": "share rota",
        "top_company_out_share": "share viação",
    }
    # fallback: usa o próprio nome
    for c in used_cols:
        aliases.setdefault(c, c)
    return aliases


def build_cluster_names(
    centroid_labels: pd.DataFrame,
    used_cols: List[str],
    top_vars_for_name: Optional[List[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
    max_fragments: int = 3,
) -> pd.DataFrame:
    """
    Gera um nome legível por cluster a partir dos rótulos por variável.
    Estratégia simples: escolhe variáveis prioritárias e concatena fragmentos.
    """
    aliases = aliases or default_aliases(used_cols)
    if top_vars_for_name is None:
        # prioridade sugerida: frequência, rotas, preço
        prio = [v for v in ["frequency_12m", "n_unique_routes_out", "avg_price_per_ticket",
                            "top_route_out_share", "top_company_out_share"] if v in used_cols]
    else:
        prio = [v for v in top_vars_for_name if v in used_cols]

    rows = []
    for _, row in centroid_labels.iterrows():
        cid = int(row["cluster_id"])
        frags: List[str] = []
        for v in prio:
            lab = row.get(f"{v}_label", "Médio")
            frag = summarize_label(lab, aliases.get(v, v))
            frags.append(frag)
        # mantém no máx. N fragmentos
        name = " · ".join(frags[:max_fragments])
        rows.append({"cluster_id": cid, "cluster_name": name})
    return pd.DataFrame(rows)


# -----------------------------
# Pipeline + salvamento
# -----------------------------
def run_labeling_pipeline(
    clustering_dir: Path | str,
    scaler_path: Path | str,
    used_cols: List[str],
    low_thr: float = -0.5,
    high_thr: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Orquestra:
      - carrega artefatos
      - calcula médias globais (via scaler.mean_)
      - rotula centróides em {Baixo,Médio,Alto}
      - cria nomes legíveis por cluster
      - retorna DataFrames
    """
    assign, cz, cu, scaler = load_artifacts(clustering_dir, scaler_path)
    means = global_means_from_scaler(scaler, used_cols)

    centroid_labels = build_centroid_labels(
        centroids_z=cz,
        used_cols=used_cols,
        low_thr=low_thr,
        high_thr=high_thr,
    )
    cluster_names = build_cluster_names(centroid_labels, used_cols)

    # resumo final (junta labels + valores unstd para interpretação)
    # garante correspondência de ordem por cluster_id
    summary = cz.merge(cu, on="cluster_id", how="left").merge(centroid_labels, on="cluster_id", how="left")

    return {
        "global_means": means.to_frame(),  # (col -> global_mean)
        "centroid_labels": centroid_labels,  # cluster_id + *_label
        "cluster_names": cluster_names,      # cluster_id + cluster_name
        "centroid_summary": summary,         # centróides z + unstd + labels
        "assignments": assign,               # só reexposto para conveniência
    }


def save_labeling_outputs(
    outputs: Dict[str, pd.DataFrame],
    out_dir: Path | str,
) -> None:
    """
    Salva:
      - global_means.csv
      - centroid_labels.parquet
      - cluster_names.parquet
      - centroid_summary.parquet
      - assignments_with_names.parquet
    """
    outdir = ensure_dir(out_dir)

    # 1) médias globais (csv p/ facilitar leitura rápida)
    outputs["global_means"].to_csv(Path(outdir) / "global_means.csv")

    # 2) labels, nomes, resumo
    save_parquet(outputs["centroid_labels"], Path(outdir) / "centroid_labels.parquet")
    save_parquet(outputs["cluster_names"], Path(outdir) / "cluster_names.parquet")
    save_parquet(outputs["centroid_summary"], Path(outdir) / "centroid_summary.parquet")

    # 3) assignments + nomes (útil para análises posteriores)
    ass_named = outputs["assignments"].merge(outputs["cluster_names"], on="cluster_id", how="left")
    save_parquet(ass_named, Path(outdir) / "cluster_assignments_named.parquet")

    logging.info("Rótulos e nomes salvos em: %s", outdir)