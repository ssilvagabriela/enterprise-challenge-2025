# src/cluster_trainer.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from src.io_saver import ensure_dir, save_single_parquet
from src import schema_config as sc

logger = logging.getLogger(__name__)


# =====================================================================
# Model factory
# =====================================================================
def _make_model(
    k: int,
    *,
    algorithm: str = "kmeans",
    random_state: int = 42,
    n_init: int | str = 10,
    max_iter: int = 300,
    batch_size: int = 1024,
):
    """
    Constrói o estimador de clustering conforme o algoritmo solicitado.
    """
    algo = algorithm.lower()
    if algo in {"kmeans", "k-means"}:
        return KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    elif algo in {"minibatchkmeans", "minibatch-kmeans", "mbkmeans"}:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init if isinstance(n_init, int) else 10,
            max_iter=max_iter,
            batch_size=batch_size,
        )
    raise ValueError(f"Algoritmo não suportado: {algorithm!r}. Use 'kmeans' ou 'minibatchkmeans'.")


# =====================================================================
# Avaliação de k
# =====================================================================
@dataclass
class EvalResult:
    k: int
    inertia: float
    silhouette: float


def evaluate_k_range(
    X: np.ndarray,
    *,
    k_values: List[int],
    algorithm: str = "kmeans",
    random_state: int = 42,
    n_init: int | str = 10,
    max_iter: int = 300,
    batch_size: int = 1024,
    silhouette_max_samples: int = 10000,
) -> List[EvalResult]:
    """
    Treina modelos para cada k e retorna lista de métricas (inertia, silhouette).
    - Silhouette é calculado em amostra aleatória (até `silhouette_max_samples`) para velocidade.
    - k inviáveis são ignorados com warning (ex.: k > n amostras únicas).
    """
    n, d = X.shape
    rng = np.random.RandomState(random_state)

    results: List[EvalResult] = []
    for k in sorted(set(k_values)):
        if k < 2 or k > n:
            logger.warning("k=%d inválido para n=%d. Pulando.", k, n)
            continue

        model = _make_model(
            k,
            algorithm=algorithm,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
        )

        try:
            labels = model.fit_predict(X)
            inertia = float(model.inertia_)
        except Exception as e:
            logger.warning("Falha ao treinar k=%d (%s). Pulando este k.", k, e)
            continue

        # Silhouette somente se houver >1 cluster e ambos com amostras
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            sil = np.nan
        else:
            if n > silhouette_max_samples:
                idx = rng.choice(n, size=silhouette_max_samples, replace=False)
                sil = silhouette_score(X[idx], labels[idx], metric="euclidean")
            else:
                sil = silhouette_score(X, labels, metric="euclidean")

        results.append(EvalResult(k=k, inertia=inertia, silhouette=float(sil)))
        logger.info("k=%d | inertia=%.4f | silhouette=%s", k, inertia, f"{sil:.4f}" if np.isfinite(sil) else "nan")

    if not results:
        raise RuntimeError("Nenhum k válido pôde ser avaliado.")
    return results


def _choose_k(
    evals: List[EvalResult],
) -> int:
    """
    Escolha de k:
      1) Maior silhouette (desempate por menor inércia).
      2) Se todos NaN, fallback "cotovelo": maior queda relativa de inércia.
    """
    df = pd.DataFrame([e.__dict__ for e in evals]).sort_values("k").reset_index(drop=True)

    # 1) Por silhouette (se houver pelo menos um valor finito)
    df_sil = df[np.isfinite(df["silhouette"])]
    if not df_sil.empty:
        max_sil = df_sil["silhouette"].max()
        cand = df_sil[df_sil["silhouette"] == max_sil].sort_values("inertia")
        k_star = int(cand.iloc[0]["k"])
        return k_star

    # 2) Fallback "cotovelo": maior queda relativa de inércia
    #    Δ_i = (inertia_{i-1} - inertia_i) / inertia_{i-1}
    inertia = df["inertia"].values
    ks = df["k"].values
    if len(inertia) < 2:
        return int(ks[0])
    drops = (inertia[:-1] - inertia[1:]) / np.clip(inertia[:-1], 1e-12, None)
    i_star = int(np.argmax(drops))
    return int(ks[i_star + 1])


# =====================================================================
# Treino final e centróides
# =====================================================================
def fit_final_model(
    X: np.ndarray,
    *,
    k: int,
    algorithm: str = "kmeans",
    random_state: int = 42,
    n_init: int | str = 10,
    max_iter: int = 300,
    batch_size: int = 1024,
):
    """
    Treina o modelo final com k definido.
    """
    model = _make_model(
        k,
        algorithm=algorithm,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
    )
    labels = model.fit_predict(X)
    return model, labels


def centroids_z_and_unstd(
    model,
    *,
    scaler,          # StandardScaler treinado
    used_cols: List[str],  # ordem das colunas no treino/padronização
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrai centróides no espaço z (padronizado) e no espaço original (despadronizado).
    Usa inverse_transform do scaler; em falha, cai para reconstrução manual.
    """
    Zc = np.asarray(model.cluster_centers_, dtype=float)  # (k, p)
    df_z = pd.DataFrame(Zc, columns=[f"{c}_z" for c in used_cols])

    try:
        uc = scaler.inverse_transform(Zc)
    except Exception as e:
        logger.warning("Falha no inverse_transform do scaler (%s). Usando reconstrução manual.", e)
        # fallback manual: x = z*scale + mean
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if mean is None or scale is None:
            raise RuntimeError("Scaler sem atributos mean_/scale_ para reconstrução manual.")
        uc = Zc * scale[np.newaxis, :] + mean[np.newaxis, :]

    df_unstd = pd.DataFrame(uc, columns=used_cols)
    return df_z, df_unstd


# =====================================================================
# Pipeline de clustering (alto nível)
# =====================================================================
def run_clustering_pipeline(
    df_std: Optional[pd.DataFrame] = None,
    *,
    X: Optional[np.ndarray] = None,            # opcional: matriz já pronta
    id_col: str = getattr(sc, "CONTACT_ID", "fk_contact"),
    used_cols: Optional[List[str]] = None,     # obrigatória se X for None? -> será inferida de df_std
    z_suffix: str = "_z",

    k_values: List[int] = list(range(2, 11)),
    algorithm: str = "kmeans",
    random_state: int = 42,
    n_init: int | str = 10,
    max_iter: int = 300,
    batch_size: int = 1024,
    silhouette_max_samples: int = 10000,
) -> Dict[str, object]:
    """
    Executa avaliação de k, escolhe k*, treina modelo final e retorna artefatos:
      - assignments (DataFrame: id_col, cluster_id)
      - centroids_z (DataFrame)
      - centroids_unstd (DataFrame)
      - evals (DataFrame com métricas por k e metadados)
      - model (objeto KMeans/MiniBatchKMeans)
      - k_best (int)
      - meta (dict)
    """
    if X is None:
        if df_std is None:
            raise ValueError("Forneça df_std (com colunas *_z) ou X (np.ndarray) já pronto.")
        if used_cols is None:
            # inferir pelas colunas padronizadas do df_std
            used_cols = [c[:-len(z_suffix)] for c in df_std.columns if c.endswith(z_suffix)]
            if not used_cols:
                raise ValueError("Nenhuma coluna *_z encontrada para compor X. Informe 'used_cols' ou 'X'.")
        z_cols = [f"{c}{z_suffix}" for c in used_cols]
        missing = [c for c in z_cols if c not in df_std.columns]
        if missing:
            raise ValueError(f"Colunas padronizadas ausentes no df_std: {missing}")
        X = df_std[z_cols].values
    else:
        if used_cols is None:
            raise ValueError("Quando X é fornecido, 'used_cols' deve ser informado (ordem do treino).")

    # 1) Avaliar k
    evals = evaluate_k_range(
        X,
        k_values=k_values,
        algorithm=algorithm,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
        silhouette_max_samples=silhouette_max_samples,
    )
    k_best = _choose_k(evals)
    logger.info("k* escolhido: %d", k_best)

    # 2) Treinar final
    model, labels = fit_final_model(
        X,
        k=k_best,
        algorithm=algorithm,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
    )

    # 3) Centrôides nos dois espaços
    # ✅ usa apenas um keyword 'scaler'
    scaler_arg = None if df_std is None else getattr(df_std, "scaler", None)
    df_cz, df_cu = centroids_z_and_unstd(
        model,
        used_cols=used_cols,
        scaler=scaler_arg,
    )

    # OBS: acima tentamos pegar scaler a partir de df_std.scaler se o chamador anexou;
    # se não existir, o chamador deve reconstruir mais adiante no artifacts. Como
    # este módulo não recebe o scaler diretamente aqui, oferecemos também a função abaixo
    # (centroids_com_scaler) para reconstrução explícita com scaler fornecido.

    # Como df_std não carrega o scaler por padrão, em muitos fluxos preferimos
    # pedir explicitamente o scaler para reconstruir centróides. Então, oferecemos:
    assignments = None
    if df_std is not None and id_col in df_std.columns:
        assignments = df_std[[id_col]].copy()
        assignments["cluster_id"] = labels.astype(int)
    else:
        # Fallback: índice sequencial
        assignments = pd.DataFrame({id_col: np.arange(len(labels)), "cluster_id": labels.astype(int)})

    # Tabela de métricas
    df_evals = pd.DataFrame(
        [{"k": e.k, "inertia": e.inertia, "silhouette": e.silhouette} for e in evals]
    )
    df_evals["algorithm"] = algorithm
    df_evals["random_state"] = random_state
    df_evals["is_final"] = df_evals["k"] == k_best

    meta = {
        "algorithm": algorithm,
        "random_state": random_state,
        "k_values": k_values,
        "k_best": k_best,
        "used_cols": used_cols,
    }

    artifacts = {
        "assignments": assignments,
        "centroids_z": df_cz,
        "centroids_unstd": df_cu,  # pode vir vazio se não for possível reconstruir (ver nota acima)
        "evals": df_evals,
        "model": model,
        "k_best": k_best,
        "meta": meta,
    }
    return artifacts


# Versão explícita quando o SCALER é fornecido (recomendada para reconstruir centróides)
def centroids_com_scaler(
    model,
    *,
    scaler,
    used_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper explícito para reconstruir centróides com um StandardScaler conhecido.
    """
    return centroids_z_and_unstd(model, scaler=scaler, used_cols=used_cols)


# =====================================================================
# I/O de artefatos
# =====================================================================
def save_clustering_artifacts(
    artifacts: Dict[str, object],
    outdir: Path | str,
) -> Dict[str, Path]:
    """
    Salva assignments, centróides e métricas em formato consistente:
      - cluster_assignments.parquet
      - cluster_centroids_z.parquet
      - cluster_centroids_unstd.parquet
      - k_selection_metrics.csv
      - k_selection_metrics.parquet
    Retorna os caminhos salvos.
    """
    outdir = Path(outdir)
    ensure_dir(outdir)

    paths: Dict[str, Path] = {}

    # Assignments
    if isinstance(artifacts.get("assignments"), pd.DataFrame):
        paths["assignments_parquet"] = save_single_parquet(
            artifacts["assignments"], outdir / "cluster_assignments.parquet"
        )

    # Centroids Z
    if isinstance(artifacts.get("centroids_z"), pd.DataFrame):
        paths["centroids_z_parquet"] = save_single_parquet(
            artifacts["centroids_z"], outdir / "cluster_centroids_z.parquet"
        )

    # Centroids Unstd
    if isinstance(artifacts.get("centroids_unstd"), pd.DataFrame):
        paths["centroids_unstd_parquet"] = save_single_parquet(
            artifacts["centroids_unstd"], outdir / "cluster_centroids_unstd.parquet"
        )

    # Métricas (CSV + Parquet)
    if isinstance(artifacts.get("evals"), pd.DataFrame):
        df_evals: pd.DataFrame = artifacts["evals"]
        csv_path = outdir / "k_selection_metrics.csv"
        df_evals.to_csv(csv_path, index=False)
        paths["metrics_csv"] = csv_path

        try:
            paths["metrics_parquet"] = save_single_parquet(
                df_evals, outdir / "k_selection_metrics.parquet"
            )
        except Exception as e:
            logger.warning("Falha ao salvar métricas em Parquet (%s). Mantendo apenas CSV.", e)

    return paths


# =====================================================================
# Leitura de features padronizadas (útil para pipelines externos)
# =====================================================================
def load_standardized_features(
    parquet_path: Path | str,
    *,
    id_col: str = getattr(sc, "CONTACT_ID", "fk_contact"),
    z_suffix: str = "_z",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lê o Parquet de features padronizadas e retorna:
      - df_std
      - used_cols (no espaço original, isto é, sem o sufixo _z)
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {parquet_path}")

    df_std = pd.read_parquet(parquet_path, engine="pyarrow")
    if id_col not in df_std.columns:
        logger.warning("Coluna de id '%s' não encontrada. Prosseguindo sem id explícito.", id_col)

    used_cols = [c[:-len(z_suffix)] for c in df_std.columns if c.endswith(z_suffix)]
    if not used_cols:
        raise ValueError(f"Nenhuma coluna '*{z_suffix}' encontrada em {parquet_path.name}.")

    return df_std, used_cols
