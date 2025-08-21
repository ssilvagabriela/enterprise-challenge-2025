from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Literal, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import schema_config as sc
from src.io_saver import ensure_dir, save_parquet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ClusterEval:
    k: int
    inertia: float
    silhouette: Optional[float]


def _get_algo(
    algo: Literal["kmeans", "minibatch"],
    k: int,
    random_state: int = 42,
    n_init: int = 10,
    batch_size: int = 2048,
    max_iter: int = 300,
):
    if algo == "kmeans":
        return KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        )
    elif algo == "minibatch":
        return MiniBatchKMeans(
            n_clusters=k,
            n_init=n_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        )
    else:
        raise ValueError("algo deve ser 'kmeans' ou 'minibatch'.")


def _maybe_sample_indices(n: int, max_samples: int) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(42)
    return rng.choice(n, size=max_samples, replace=False)


def evaluate_k_range(
    X: np.ndarray,
    k_values: Iterable[int] = range(2, 11),
    algo: Literal["kmeans", "minibatch"] = "minibatch",
    random_state: int = 42,
    silhouette_max_samples: int = 20000,
) -> List[ClusterEval]:
    """
    Treina modelos para vários k e retorna SSE (inertia) e Silhouette.
    Para Silhouette, amostra no máximo 'silhouette_max_samples' pontos por performance.
    """
    n = X.shape[0]
    idx_s = _maybe_sample_indices(n, silhouette_max_samples)

    evals: List[ClusterEval] = []
    for k in k_values:
        model = _get_algo(algo, k, random_state=random_state)
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        sil = None
        # Silhouette só faz sentido com >1 cluster e >1 amostra no slice
        try:
            sil = float(silhouette_score(X[idx_s], labels[idx_s], metric="euclidean"))
        except Exception as e:
            logger.warning(f"Silhouette falhou para k={k}: {e!r}")
            sil = None

        evals.append(ClusterEval(k=k, inertia=inertia, silhouette=sil))
        logger.info("k=%d | inertia=%.4f | silhouette=%s", k, inertia, f"{sil:.4f}" if sil is not None else "NA")
    return evals


def pick_best_k(
    evals: List[ClusterEval],
    prefer: Literal["silhouette", "elbow"] = "silhouette",
    default_k: int = 5,
) -> int:
    """
    Heurística simples:
    - se 'silhouette': pega k com maior silhouette (desempate pelo menor inertia)
    - se 'elbow': escolhe o joelho via razão sucessiva de quedas (simples)
    - fallback: default_k
    """
    # 1) silhouette
    if prefer == "silhouette":
        sil = [(e.k, e.silhouette, e.inertia) for e in evals if e.silhouette is not None]
        if sil:
            # maior silhouette, em empate pega menor inertia
            sil.sort(key=lambda t: (-t[1], t[2]))
            return int(sil[0][0])

    # 2) elbow (queda relativa da inércia)
    if evals and prefer == "elbow":
        evals_sorted = sorted(evals, key=lambda e: e.k)
        inertias = np.array([e.inertia for e in evals_sorted], dtype=float)
        ks = np.array([e.k for e in evals_sorted])
        # queda relativa: (I_{k-1} - I_k)/I_{k-1}
        drops = (inertias[:-1] - inertias[1:]) / np.maximum(inertias[:-1], 1e-12)
        # escolhe k no ponto de maior queda (i -> k = ks[i+1])
        if len(drops) > 0:
            best_i = int(np.argmax(drops))
            return int(ks[best_i + 1])

    # fallback
    logger.warning("Não foi possível decidir k automaticamente. Usando default_k=%d.", default_k)
    return int(default_k)


def train_final_model(
    X: np.ndarray,
    k: int,
    algo: Literal["kmeans", "minibatch"] = "minibatch",
    random_state: int = 42,
) -> Tuple[np.ndarray, object]:
    model = _get_algo(algo, k, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model


def centroids_z_and_unstd(
    model,
    scaler: StandardScaler,
    used_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna centróides em z-score (Z) e despadronizados (Unstd).
    - Para StandardScaler: X_unstd = Z * scale_ + mean_
    """
    if not hasattr(model, "cluster_centers_"):
        raise ValueError("Modelo não possui cluster_centers_.")
    Z = np.asarray(model.cluster_centers_, dtype=float)  # (k, p)

    # reconstroi no espaço original
    if not isinstance(scaler, StandardScaler):
        raise TypeError("scaler deve ser StandardScaler compatível com etapa 5.")
    if getattr(scaler, "mean_", None) is None or getattr(scaler, "scale_", None) is None:
        raise ValueError("Scaler não possui mean_ e scale_. Use o scaler salvo na etapa 5.")

    mean = scaler.mean_.astype(float)
    scale = scaler.scale_.astype(float)
    X_unstd = Z * scale + mean

    zcols = [f"{c}_z" for c in used_cols]
    uncols = [f"{c}_unstd" for c in used_cols]

    dfZ = pd.DataFrame(Z, columns=zcols)
    dfU = pd.DataFrame(X_unstd, columns=uncols)

    dfZ.insert(0, "cluster_id", np.arange(Z.shape[0], dtype=int))
    dfU.insert(0, "cluster_id", np.arange(Z.shape[0], dtype=int))
    return dfZ, dfU


def load_standardized_features(path_parquet: Path | str, id_col: str) -> pd.DataFrame:
    p = Path(path_parquet)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    df = pd.read_parquet(p)
    if id_col not in df.columns:
        raise ValueError(f"Coluna de id '{id_col}' não encontrada em {p}.")
    return df


def load_scaler(path_joblib: Path | str) -> StandardScaler:
    p = Path(path_joblib)
    if not p.exists():
        raise FileNotFoundError(f"Scaler não encontrado: {p}")
    scaler = joblib.load(p)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Objeto carregado não é StandardScaler.")
    return scaler


def run_clustering_pipeline(
    df_std: pd.DataFrame,
    used_cols: List[str],
    scaler: StandardScaler,
    algo: Literal["kmeans", "minibatch"] = "minibatch",
    k_values: Iterable[int] = range(2, 11),
    prefer: Literal["silhouette", "elbow"] = "silhouette",
    default_k: int = 5,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame | object | int | List[ClusterEval]]:
    """
    Orquestra:
      - avalia k
      - escolhe k
      - treina final
      - devolve assignments e centróides (Z e unstd) + métricas.
    """
    # seleciona matriz X (nas colunas padronizadas *_z)
    zcols = [f"{c}_z" for c in used_cols if f"{c}_z" in df_std.columns]
    if not zcols:
        raise ValueError("Nenhuma coluna *_z encontrada no DataFrame padronizado.")
    X = df_std[zcols].to_numpy(dtype=float, copy=False)

    # 1) avaliar k
    evals = evaluate_k_range(
        X=X,
        k_values=k_values,
        algo=algo,
        random_state=random_state,
    )

    # 2) escolher k
    k_best = pick_best_k(evals, prefer=prefer, default_k=default_k)
    logger.info("k escolhido: %d", k_best)

    # 3) treinar final
    labels, model = train_final_model(
        X=X, k=k_best, algo=algo, random_state=random_state
    )

    # 4) assignments
    assign = df_std[[sc.CONTACT_ID]].copy()
    assign["cluster_id"] = labels.astype(int)

    # 5) centróides (z e despadronizados)
    dfZ, dfU = centroids_z_and_unstd(model, scaler, used_cols=used_cols)

    # 6) métricas em DataFrame
    df_evals = pd.DataFrame(
        [{"k": e.k, "inertia": e.inertia, "silhouette": e.silhouette} for e in evals]
    )

    return {
        "k_best": k_best,
        "model": model,
        "assignments": assign,
        "centroids_z": dfZ,
        "centroids_unstd": dfU,
        "evals": df_evals,
    }


def save_clustering_artifacts(
    artifacts: Dict[str, pd.DataFrame | object | int],
    output_dir: Path | str,
) -> None:
    """
    Salva:
      - assignments.parquet
      - centroids_z.parquet
      - centroids_unstd.parquet
      - k_selection.csv
      - modelo.joblib
    """
    outdir = ensure_dir(output_dir)
    # tabelas
    save_parquet(artifacts["assignments"], outdir / "cluster_assignments.parquet")
    save_parquet(artifacts["centroids_z"], outdir / "cluster_centroids_z.parquet")
    save_parquet(artifacts["centroids_unstd"], outdir / "cluster_centroids_unstd.parquet")
    artifacts["evals"].to_csv(outdir / "k_selection_metrics.csv", index=False)
    # modelo
    joblib.dump(artifacts["model"], outdir / "kmeans_model.joblib")
    logger.info("Artefatos de clustering salvos em: %s", outdir)