# src/cluster_labeler.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.io_saver import save_single_parquet, ensure_dir

logger = logging.getLogger(__name__)


# =========================================================
# Utilidades de leitura
# =========================================================
def _check_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Arquivo '{label}' não encontrado: {p}")


def read_parquet_safe(path: Path) -> pd.DataFrame:
    _check_exists(path, path.name)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Lido %s (%d linhas, %d colunas)", path, df.shape[0], df.shape[1])
    return df


# =========================================================
# Rotulagem por z-score
# =========================================================
def z_to_label(z: float, low_thr: float = -0.5, high_thr: float = 0.5) -> str:
    """
    Converte um valor z em {Baixo, Médio, Alto} usando thresholds inclusivos.
    - z <= low_thr   -> 'Baixo'
    - low_thr < z < high_thr -> 'Médio'
    - z >= high_thr  -> 'Alto'
    """
    if pd.isna(z):
        return "Médio"
    if z <= low_thr:
        return "Baixo"
    if z >= high_thr:
        return "Alto"
    return "Médio"


def build_centroid_labels(
    centroids_z: pd.DataFrame,
    *,
    used_cols: List[str],
    cluster_id_col: str = "cluster_id",
    low_thr: float = -0.5,
    high_thr: float = 0.5,
) -> pd.DataFrame:
    """
    Gera tabela longa com rótulos por variável/cluster.
    Espera que centroids_z possua colunas <feature>_z e 'cluster_id' (ou index).
    Retorna colunas: [cluster_id, feature, z, label]
    """
    df = centroids_z.copy()

    # cluster_id: do index ou de coluna
    if cluster_id_col not in df.columns:
        df = df.reset_index().rename(columns={"index": cluster_id_col})

    # coleta colunas z e checa presença
    z_cols = [f"{c}_z" for c in used_cols]
    missing = [c for c in z_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas z ausentes em centroids_z: {missing}")

    # monta formato longo
    long = df[[cluster_id_col] + z_cols].melt(
        id_vars=[cluster_id_col],
        var_name="feature_z",
        value_name="z",
    )
    long["feature"] = long["feature_z"].str.replace(r"_z$", "", regex=True)
    long["label"] = long["z"].apply(lambda v: z_to_label(v, low_thr=low_thr, high_thr=high_thr)).astype("string")

    # tipos estáveis
    long[cluster_id_col] = long[cluster_id_col].astype(int)
    long["feature"] = long["feature"].astype("string")
    return long[[cluster_id_col, "feature", "z", "label"]]


# =========================================================
# Nomes legíveis de cluster (aliases + prioridade)
# =========================================================
_DEFAULT_ALIASES = {
    "frequency_12m": "Frequência",
    "n_unique_routes_out": "Rotas",
    "avg_price_per_ticket": "Preço Médio",
    "top_route_out_share": "Foco na Rota",
    "top_company_out_share": "Foco na Viação",
    "pct_round_trip": "% Ida/Volta",
    "monetary_gmv_12m": "GMV",
    "total_tickets_12m": "Tickets",
    # Sazonalidade (exemplos mais comuns)
    "pct_weekend": "% Fim‑de‑semana",
    "pct_q1": "% Q1", "pct_q2": "% Q2", "pct_q3": "% Q3", "pct_q4": "% Q4",
    "pct_verao": "% Verão", "pct_outono": "% Outono", "pct_inverno": "% Inverno", "pct_primavera": "% Primavera",
    "pct_madrugada": "% Madrugada", "pct_manha": "% Manhã", "pct_tarde": "% Tarde", "pct_noite": "% Noite",
    "pct_holiday": "% Feriados",
}

_DEFAULT_PRIORITY = [
    # principais para segmentação
    "frequency_12m", "n_unique_routes_out", "avg_price_per_ticket",
    "top_route_out_share", "top_company_out_share", "pct_round_trip",
    # intensidade
    "total_tickets_12m", "monetary_gmv_12m",
    # sazonalidade
    "pct_weekend", "pct_q3", "pct_q1", "pct_q2", "pct_q4",
    "pct_verao", "pct_outono", "pct_inverno", "pct_primavera",
    "pct_noite", "pct_tarde", "pct_manha", "pct_madrugada",
    "pct_holiday",
]


def _label_fragment(var: str, label: str, aliases: Dict[str, str]) -> str:
    """
    Constrói fragmento amigável para nome do cluster.
    - Para variáveis com sentido de intensidade/compartilhamento, deixa explícito 'alto/baixo'.
    """
    base = aliases.get(var, var)
    if label == "Alto":
        return f"Alto {base}"
    if label == "Baixo":
        return f"Baixo {base}"
    return f"Médio {base}"


def build_cluster_names(
    centroid_labels: pd.DataFrame,
    *,
    used_cols: List[str],
    cluster_id_col: str = "cluster_id",
    aliases: Optional[Dict[str, str]] = None,
    priority: Optional[List[str]] = None,
    max_fragments: int = 3,
) -> pd.DataFrame:
    """
    A partir dos rótulos por variável, compõe nomes legíveis e determinísticos.
    Retorna [cluster_id, cluster_name]
    """
    aliases = {**_DEFAULT_ALIASES, **(aliases or {})}
    priority = priority or _DEFAULT_PRIORITY

    # Ordena deterministicamente por cluster_id e prioridade de variáveis
    df = centroid_labels.copy()
    df[cluster_id_col] = df[cluster_id_col].astype(int)

    # apenas variáveis usadas
    df = df[df["feature"].isin(used_cols)].copy()

    # ordem de prioridade fixa: presente -> posição
    prio_rank = {v: i for i, v in enumerate(priority)}
    df["__prio__"] = df["feature"].map(lambda v: prio_rank.get(v, len(priority)))

    # ordenar por cluster e por prioridade
    df = df.sort_values([cluster_id_col, "__prio__", "feature"]).reset_index(drop=True)

    names = []
    for cid, part in df.groupby(cluster_id_col, sort=True):
        # pega apenas os primeiros 'max_fragments' rótulos distintos pela prioridade
        frags = []
        seen_feats = set()
        for _, row in part.iterrows():
            feat = row["feature"]
            if feat in seen_feats:
                continue
            frag = _label_fragment(feat, str(row["label"]), aliases)
            frags.append(frag)
            seen_feats.add(feat)
            if len(frags) >= max_fragments:
                break
        cluster_name = " | ".join(frags) if frags else "Cluster"
        names.append((int(cid), cluster_name))

    out = pd.DataFrame(names, columns=[cluster_id_col, "cluster_name"])
    out["cluster_name"] = out["cluster_name"].astype("string")
    return out


# =========================================================
# Resumo dos centróides + labels (+ metadados)
# =========================================================
def build_centroid_summary(
    centroids_z: pd.DataFrame,
    centroid_labels: pd.DataFrame,
    *,
    centroids_unstd: Optional[pd.DataFrame] = None,
    cluster_id_col: str = "cluster_id",
    meta: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Monta uma visão 'wide' com z-scores, valores originais (se disponíveis),
    e rótulos por variável. Inclui metadados opcionais como colunas extras.
    """
    cz = centroids_z.copy()
    if cluster_id_col not in cz.columns:
        cz = cz.reset_index().rename(columns={"index": cluster_id_col})

    # labels: pivot para colunas por variável
    lab_wide = centroid_labels.pivot(index=cluster_id_col, columns="feature", values="label")
    lab_wide.columns = [f"label__{c}" for c in lab_wide.columns]
    lab_wide = lab_wide.reset_index()

    # merge dos z
    summary = cz.merge(lab_wide, on=cluster_id_col, how="left")

    # valores originais são opcionais
    if centroids_unstd is not None and not centroids_unstd.empty:
        cu = centroids_unstd.copy()
        if cluster_id_col not in cu.columns:
            cu = cu.reset_index().rename(columns={"index": cluster_id_col})
        summary = summary.merge(cu, on=cluster_id_col, how="left", suffixes=("", "_orig"))

    # metadados opcionais como colunas (simples)
    if meta:
        for k, v in meta.items():
            if isinstance(v, (str, int, float, np.number, pd.Timestamp)):
                summary[k] = v

    # tipos estáveis
    summary[cluster_id_col] = summary[cluster_id_col].astype(int)
    return summary


# =========================================================
# Pipeline principal
# =========================================================
def run_labeling_pipeline(
    *,
    centroids_z: pd.DataFrame,
    used_cols: List[str],
    assignments: Optional[pd.DataFrame] = None,
    centroids_unstd: Optional[pd.DataFrame] = None,
    metrics: Optional[pd.DataFrame] = None,
    cluster_id_col: str = "cluster_id",
    id_col: str = "fk_contact",
    low_thr: float = -0.5,
    high_thr: float = 0.5,
    max_fragments: int = 3,
    aliases: Optional[Dict[str, str]] = None,
    priority: Optional[List[str]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Orquestra rotulagem, nomes e resumo. Retorna dict com:
      - centroid_labels (long)
      - cluster_names
      - centroid_summary
      - assignments_named (se 'assignments' for fornecido)
    """
    # 1) labels
    centroid_labels = build_centroid_labels(
        centroids_z,
        used_cols=used_cols,
        cluster_id_col=cluster_id_col,
        low_thr=low_thr,
        high_thr=high_thr,
    )

    # 2) nomes (determinísticos)
    cluster_names = build_cluster_names(
        centroid_labels,
        used_cols=used_cols,
        cluster_id_col=cluster_id_col,
        aliases=aliases,
        priority=priority,
        max_fragments=max_fragments,
    )

    # 3) resumo (cz + cu opcional + labels + meta)
    meta_all = dict(meta or {})
    if metrics is not None and not metrics.empty:
        # tenta extrair metadados úteis
        if "algorithm" in metrics.columns:
            meta_all.setdefault("algorithm", str(metrics["algorithm"].iloc[0]))
        if "k" in metrics.columns:
            # se tabela tiver várias linhas, usar o k marcado como final (ou o modo mais frequente)
            if "is_final" in metrics.columns and metrics["is_final"].any():
                meta_all.setdefault("k", int(metrics.loc[metrics["is_final"], "k"].iloc[0]))
            else:
                meta_all.setdefault("k", int(metrics["k"].mode().iloc[0]))
        if "random_state" in metrics.columns:
            meta_all.setdefault("random_state", int(metrics["random_state"].iloc[0]))

    centroid_summary = build_centroid_summary(
        centroids_z=centroids_z,
        centroid_labels=centroid_labels,
        centroids_unstd=centroids_unstd,
        cluster_id_col=cluster_id_col,
        meta=meta_all or None,
    )

    # 4) assignments nomeados (opcional)
    assignments_named = pd.DataFrame()
    if assignments is not None and not assignments.empty:
        if cluster_id_col not in assignments.columns:
            raise KeyError(f"assignments sem coluna '{cluster_id_col}'.")
        if id_col not in assignments.columns:
            logger.warning("assignments sem id '%s'. Usando índice como id.", id_col)
            assignments = assignments.reset_index().rename(columns={"index": id_col})
        assignments_named = assignments.merge(cluster_names, on=cluster_id_col, how="left")
        # tipagem estável
        assignments_named[cluster_id_col] = assignments_named[cluster_id_col].astype(int)
        assignments_named[id_col] = assignments_named[id_col].astype("string")

    return {
        "centroid_labels": centroid_labels,
        "cluster_names": cluster_names,
        "centroid_summary": centroid_summary,
        "assignments_named": assignments_named,
    }


# =========================================================
# I/O de artefatos
# =========================================================
def load_artifacts(
    *,
    centroids_z_path: Path,
    used_cols: List[str],
    assignments_path: Optional[Path] = None,
    centroids_unstd_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    cluster_id_col: str = "cluster_id",
    id_col: str = "fk_contact",
) -> Dict[str, pd.DataFrame]:
    """
    Lê artefatos gerados pelo cluster_trainer/artifacts:
      - centroids_z (obrigatório)
      - centroids_unstd (opcional)
      - assignments (opcional)
      - metrics (opcional)
    """
    cz = read_parquet_safe(centroids_z_path)
    if cluster_id_col not in cz.columns:
        cz = cz.reset_index().rename(columns={"index": cluster_id_col})

    cu = pd.DataFrame()
    if centroids_unstd_path is not None and centroids_unstd_path.exists():
        cu = read_parquet_safe(centroids_unstd_path)
        if cluster_id_col not in cu.columns:
            cu = cu.reset_index().rename(columns={"index": cluster_id_col})

    asg = pd.DataFrame()
    if assignments_path is not None and assignments_path.exists():
        asg = read_parquet_safe(assignments_path)

    metrics = pd.DataFrame()
    if metrics_path is not None and metrics_path.exists():
        # pode ser CSV ou Parquet
        if metrics_path.suffix.lower() == ".csv":
            _check_exists(metrics_path, "k_selection_metrics.csv")
            metrics = pd.read_csv(metrics_path)
            logger.info("Lido %s (%d linhas, %d colunas)", metrics_path, metrics.shape[0], metrics.shape[1])
        else:
            metrics = read_parquet_safe(metrics_path)

    # validação leve das colunas z
    z_cols = [f"{c}_z" for c in used_cols]
    missing = [c for c in z_cols if c not in cz.columns]
    if missing:
        raise ValueError(f"As seguintes colunas z não estão em centroids_z: {missing}")

    return {"centroids_z": cz, "centroids_unstd": cu, "assignments": asg, "metrics": metrics}


def save_labeling_outputs(
    outputs: Dict[str, pd.DataFrame],
    outdir: Path | str,
    *,
    save_csv_extras: bool = True,
) -> Dict[str, Path]:
    """
    Salva:
      - centroid_labels.parquet
      - cluster_names.parquet
      - centroid_summary.parquet
      - cluster_assignments_named.parquet (se existir)
    E, opcionalmente, CSVs: centroid_summary.csv e cluster_assignments_named.csv
    """
    outdir = Path(outdir)
    ensure_dir(outdir)

    paths: Dict[str, Path] = {}

    if "centroid_labels" in outputs and isinstance(outputs["centroid_labels"], pd.DataFrame):
        paths["centroid_labels_parquet"] = save_single_parquet(outputs["centroid_labels"], outdir / "centroid_labels.parquet")

    if "cluster_names" in outputs and isinstance(outputs["cluster_names"], pd.DataFrame):
        paths["cluster_names_parquet"] = save_single_parquet(outputs["cluster_names"], outdir / "cluster_names.parquet")

    if "centroid_summary" in outputs and isinstance(outputs["centroid_summary"], pd.DataFrame):
        paths["centroid_summary_parquet"] = save_single_parquet(outputs["centroid_summary"], outdir / "centroid_summary.parquet")
        if save_csv_extras:
            csv_path = outdir / "centroid_summary.csv"
            outputs["centroid_summary"].to_csv(csv_path, index=False)
            paths["centroid_summary_csv"] = csv_path

    if "assignments_named" in outputs and isinstance(outputs["assignments_named"], pd.DataFrame) and not outputs["assignments_named"].empty:
        paths["assignments_named_parquet"] = save_single_parquet(outputs["assignments_named"], outdir / "cluster_assignments_named.parquet")
        if save_csv_extras:
            csv_path = outdir / "cluster_assignments_named.csv"
            outputs["assignments_named"].to_csv(csv_path, index=False)
            paths["assignments_named_csv"] = csv_path

    return paths
