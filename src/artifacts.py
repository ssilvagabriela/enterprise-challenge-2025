# src/artifacts.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.io_saver import ensure_dir, save_single_parquet

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers gerais
# =============================================================================
def _today_str() -> str:
    """Data 'hoje' sem timezone/hora, estável para versionamento."""
    return date.today().strftime("%Y-%m-%d")


def _first_existing(paths: List[Path]) -> Optional[Path]:
    """Retorna o primeiro Path existente na lista (ou None)."""
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _coerce_string(s: pd.Series) -> pd.Series:
    return s.astype("string")


def _coerce_int64_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# =============================================================================
# Commit artifacts — Assignments (CSV)
# =============================================================================
def build_cluster_assignments_csv(
    assignments: pd.DataFrame,
    *,
    id_col: str = "fk_contact",
    cluster_id_col: str = "cluster_id",
    extras: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Prepara um CSV com (id, cluster_id) e metadados úteis.

    - Valida presença das colunas obrigatórias
    - Tipa 'id_col' como string e 'cluster_id' como Int64
    - Acrescenta metadados simples, se fornecidos (k, algorithm, data)
    """
    if id_col not in assignments.columns or cluster_id_col not in assignments.columns:
        raise KeyError(f"Assignments precisa conter '{id_col}' e '{cluster_id_col}'.")

    out = assignments[[id_col, cluster_id_col]].copy()
    out[id_col] = _coerce_string(out[id_col])
    out[cluster_id_col] = _coerce_int64_nullable(out[cluster_id_col])

    # metadados opcionais
    extras = dict(extras or {})
    if "k" in extras:
        out["k"] = _coerce_int64_nullable(pd.Series(extras["k"], index=out.index))
    if "algorithm" in extras:
        out["algorithm"] = _coerce_string(pd.Series(str(extras["algorithm"]), index=out.index))
    if "build_date" in extras:
        out["build_date"] = _coerce_string(pd.Series(str(extras["build_date"]), index=out.index))
    if "model_version" in extras:
        out["model_version"] = _coerce_string(pd.Series(str(extras["model_version"]), index=out.index))
    if "run_date" not in out.columns:
        out["run_date"] = _coerce_string(pd.Series(_today_str(), index=out.index))

    logger.info("Assignments CSV preparado: %d linhas", len(out))
    return out


# =============================================================================
# Centróides — JSON
# =============================================================================
@dataclass
class CentroidsJSON:
    model: str
    run_date: str
    features: List[str]
    centroids_z: List[Dict[str, float]]
    centroids_unstd: Optional[List[Dict[str, float]]]
    global_means: Dict[str, float]
    global_stds: Optional[Dict[str, float]] = None
    stats: Optional[Dict[str, object]] = None


def _to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    return df.to_dict(orient="records")


def _reconstruct_unstd_from_scaler(
    centroids_z: pd.DataFrame,
    *,
    scaler,
    used_cols: List[str],
) -> pd.DataFrame:
    """
    Reconstrói centróides no espaço original usando um StandardScaler já treinado.
    Tenta inverse_transform; em falha, usa fórmula manual x = z*scale + mean.
    """
    Z = centroids_z[[f"{c}_z" for c in used_cols]].to_numpy(dtype=float, copy=False)
    try:
        U = scaler.inverse_transform(Z)
    except Exception as e:
        logger.warning("inverse_transform falhou (%s). Usando reconstrução manual mean_/scale_.", e)
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if mean is None or scale is None:
            raise RuntimeError("Scaler não possui mean_/scale_ para reconstrução manual.")
        U = Z * np.asarray(scale)[None, :] + np.asarray(mean)[None, :]
    return pd.DataFrame(U, columns=used_cols)


def build_cluster_centroids_json(
    *,
    centroids_z: pd.DataFrame,
    used_cols: List[str],
    model_name: str,
    scaler=None,  # StandardScaler opcional
    stats: Optional[Dict[str, object]] = None,
    centroids_unstd: Optional[pd.DataFrame] = None,
) -> CentroidsJSON:
    """
    Monta o objeto de centróides para serialização em JSON.
    Inclui global_means e global_stds quando o scaler é fornecido.
    """
    cz = centroids_z.copy()
    if "cluster_id" not in cz.columns:
        cz = cz.reset_index().rename(columns={"index": "cluster_id"})

    z_cols = [f"{c}_z" for c in used_cols]
    miss = [c for c in z_cols if c not in cz.columns]
    if miss:
        raise ValueError(f"Centroids_z não contém colunas: {miss}")

    # Reconstrução 'unstd' quando necessário
    cu = None
    if centroids_unstd is not None and not centroids_unstd.empty:
        cu = centroids_unstd.copy()
    elif scaler is not None:
        cu = _reconstruct_unstd_from_scaler(cz, scaler=scaler, used_cols=used_cols)

    # global_means/stds quando scaler disponível
    global_means = {}
    global_stds = None
    if scaler is not None and getattr(scaler, "mean_", None) is not None:
        global_means = {c: float(m) for c, m in zip(used_cols, scaler.mean_)}
        if getattr(scaler, "scale_", None) is not None:
            global_stds = {c: float(s) for c, s in zip(used_cols, scaler.scale_)}

    # montar objeto
    payload = CentroidsJSON(
        model=str(model_name),
        run_date=_today_str(),
        features=list(used_cols),
        centroids_z=_to_records(cz[["cluster_id"] + z_cols]),
        centroids_unstd=_to_records(cu[["cluster_id"] + used_cols]) if cu is not None and "cluster_id" in cu.columns
        else (_to_records(pd.concat([cz[["cluster_id"]], cu], axis=1)) if cu is not None else None),
        global_means=global_means,
        global_stds=global_stds,
        stats=stats or None,
    )
    return payload


def save_centroids_json(payload: CentroidsJSON, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("Centroids JSON salvo em: %s", out_path)
    return out_path


# =============================================================================
# Excel — resumo e top-N
# =============================================================================
def _rank_top(s: pd.Series, top_n: int) -> pd.Series:
    """
    Retorna top-N como 'valor (contagem)' em uma única célula separada por ' | '.
    """
    vc = s.value_counts(dropna=True).head(top_n)
    if vc.empty:
        return pd.Series({"top": ""})
    out = " | ".join([f"{k} ({v})" for k, v in vc.items()])
    return pd.Series({"top": out})


def build_cluster_summary_xlsx(
    *,
    out_path: Path | str,
    centroids_z: pd.DataFrame,
    centroid_labels: pd.DataFrame,
    assignments_named: Optional[pd.DataFrame] = None,
    centroids_unstd: Optional[pd.DataFrame] = None,
    metrics: Optional[pd.DataFrame] = None,
    glossary: Optional[pd.DataFrame] = None,
    top_n: int = 15,
) -> Path:
    """
    Gera uma planilha com múltiplas abas:
      - overview: métricas e contagens por cluster
      - centroids_z: centróides padronizados
      - centroids_unstd: centróides no espaço original (se houver)
      - labels: rótulos por variável/cluster
      - assignments_named: clientes e seus clusters (se houver)
      - top: ranking de rotas/empresas (se 'assignments_named' trouxer colunas)
      - glossary: (opcional) dicionário de features
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # Preparos e visões auxiliares
    over = pd.DataFrame()
    if metrics is not None and not metrics.empty:
        over = metrics.copy()
    if assignments_named is not None and not assignments_named.empty:
        cnt = assignments_named.groupby("cluster_id", dropna=False).size().rename("n_customers").reset_index()
        over = cnt if over.empty else over.merge(cnt, on="cluster_id", how="left")

    # Top‑N (se colunas existirem)
    top_sheet = pd.DataFrame()
    if assignments_named is not None and not assignments_named.empty:
        tops = []
        if "top_route_out" in assignments_named.columns:
            r = assignments_named.groupby("cluster_id")["top_route_out"].apply(lambda s: _rank_top(s, top_n)).reset_index()
            r["metric"] = "top_route_out"
            tops.append(r)
        if "top_company_out" in assignments_named.columns:
            r = assignments_named.groupby("cluster_id")["top_company_out"].apply(lambda s: _rank_top(s, top_n)).reset_index()
            r["metric"] = "top_company_out"
            tops.append(r)
        if tops:
            top_sheet = pd.concat(tops, ignore_index=True)

    # Escrita do Excel com engine auto
    with pd.ExcelWriter(out_path, engine=None) as writer:
        if not over.empty:
            over.to_excel(writer, sheet_name="overview", index=False)
        centroids_z.to_excel(writer, sheet_name="centroids_z", index=False)
        if centroids_unstd is not None and not centroids_unstd.empty:
            centroids_unstd.to_excel(writer, sheet_name="centroids_unstd", index=False)
        centroid_labels.to_excel(writer, sheet_name="labels", index=False)
        if assignments_named is not None and not assignments_named.empty:
            assignments_named.to_excel(writer, sheet_name="assignments", index=False)
        if not top_sheet.empty:
            top_sheet.to_excel(writer, sheet_name="top", index=False)
        if glossary is not None and not glossary.empty:
            glossary.to_excel(writer, sheet_name="glossary", index=False)

    logger.info("Resumo Excel salvo em: %s", out_path)
    return out_path


# =============================================================================
# Outros utilitários
# =============================================================================
def copy_customer_features(
    features: pd.DataFrame,
    out_path: Path | str,
) -> Path:
    """
    Persiste o DataFrame de features finais em Parquet com escrita segura.
    """
    return save_single_parquet(features, Path(out_path))


# =============================================================================
# Orquestrador (opcional)
# =============================================================================
def orchestrate_artifacts(
    *,
    out_dir: Path | str,
    # paths preferenciais (o primeiro existente será usado)
    centroids_z_paths: List[Path],
    used_cols: List[str],
    centroids_unstd_paths: Optional[List[Path]] = None,
    scaler_path: Optional[Path] = None,  # .joblib opcional para reconstrução
    metrics_paths: Optional[List[Path]] = None,
    assignments_paths: Optional[List[Path]] = None,
    # metadata
    model_name: str = "kmeans",
    meta: Optional[Dict[str, object]] = None,
    excel_top_n: int = 15,
) -> Dict[str, Path]:
    """
    Faz a leitura dos artefatos (usando o primeiro path existente de cada lista),
    monta JSON de centróides, CSV/Parquet de assignments e Excel de resumo.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # --- localizar fontes ---
    cz_path = _first_existing([Path(p) for p in centroids_z_paths])
    if cz_path is None:
        raise FileNotFoundError("Nenhum 'centroids_z' encontrado entre os paths fornecidos.")

    cu_path = _first_existing([Path(p) for p in (centroids_unstd_paths or [])])
    metrics_path = _first_existing([Path(p) for p in (metrics_paths or [])])
    asg_path = _first_existing([Path(p) for p in (assignments_paths or [])])

    # --- ler artefatos principais ---
    cz = pd.read_parquet(cz_path, engine="pyarrow")
    if "cluster_id" not in cz.columns:
        cz = cz.reset_index().rename(columns={"index": "cluster_id"})

    cu = pd.DataFrame()
    if cu_path:
        cu = pd.read_parquet(cu_path, engine="pyarrow")
        if "cluster_id" not in cu.columns:
            cu = cu.reset_index().rename(columns={"index": "cluster_id"})

    metrics = pd.DataFrame()
    if metrics_path:
        if metrics_path.suffix.lower() == ".csv":
            metrics = pd.read_csv(metrics_path)
        else:
            metrics = pd.read_parquet(metrics_path, engine="pyarrow")

    assignments = pd.DataFrame()
    if asg_path:
        assignments = pd.read_parquet(asg_path, engine="pyarrow")

    # --- carregar scaler, se fornecido ---
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        try:
            from joblib import load
            scaler = load(scaler_path)
        except Exception as e:
            logger.warning("Não foi possível carregar scaler em %s (%s). Seguiremos sem reconstrução.", scaler_path, e)

    # --- construir e salvar JSON de centróides ---
    stats = {}
    if not metrics.empty:
        # compactar algumas métricas úteis
        try:
            k_final = int(metrics.loc[metrics["is_final"], "k"].iloc[0]) if "is_final" in metrics.columns and metrics["is_final"].any() else int(metrics["k"].mode().iloc[0])
            stats.update({"k": k_final})
        except Exception:
            pass
        if "algorithm" in metrics.columns:
            stats["algorithm"] = str(metrics["algorithm"].iloc[0])
        if "random_state" in metrics.columns:
            stats["random_state"] = int(metrics["random_state"].iloc[0])

    payload = build_cluster_centroids_json(
        centroids_z=cz,
        used_cols=used_cols,
        model_name=model_name,
        scaler=scaler,
        stats=stats or None,
        centroids_unstd=cu if not cu.empty else None,
    )

    json_path = out_dir / "cluster_centroids.json"
    save_centroids_json(payload, json_path)

    # --- assignments CSV/Parquet (se houver) ---
    paths: Dict[str, Path] = {"centroids_json": json_path}

    if not assignments.empty:
        extras = dict(meta or {})
        extras.update({k: v for k, v in stats.items()})  # inclui k/algorithm/random_state se extraídos
        csv_df = build_cluster_assignments_csv(assignments, extras=extras)
        csv_path = out_dir / "cluster_assignments.csv"
        ensure_dir(csv_path.parent)
        csv_df.to_csv(csv_path, index=False)
        paths["assignments_csv"] = csv_path

        try:
            paths["assignments_parquet"] = save_single_parquet(assignments, out_dir / "cluster_assignments.parquet")
        except Exception as e:
            logger.warning("Falha ao salvar assignments em Parquet (%s). Mantendo apenas CSV.", e)

    # --- Excel de resumo (se tivermos labels e nomes em outro fluxo, podem ser passados aqui; neste orquestrador, criamos abas básicas) ---
    # Como este módulo pode ser usado isoladamente, criamos uma planilha mínima com cz, cu e metrics.
    try:
        excel_path = out_dir / "cluster_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine=None) as writer:
            cz.to_excel(writer, sheet_name="centroids_z", index=False)
            if cu is not None and not cu.empty:
                cu.to_excel(writer, sheet_name="centroids_unstd", index=False)
            if not metrics.empty:
                metrics.to_excel(writer, sheet_name="metrics", index=False)
        paths["cluster_summary_xlsx"] = excel_path
    except Exception as e:
        logger.warning("Falha ao salvar Excel de resumo (%s).", e)

    return paths
