# src/standardizer.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

from src.io_saver import save_single_parquet

logger = logging.getLogger(__name__)


# =========================================================
# Utilidades internas
# =========================================================
def _select_present_columns(df: pd.DataFrame, cluster_var: List[str]) -> Tuple[List[str], List[str]]:
    """Retorna (presentes, ausentes) preservando a ordem fornecida."""
    present = [c for c in cluster_var if c in df.columns]
    missing = [c for c in cluster_var if c not in df.columns]
    return present, missing


def _auto_log1p_columns(df: pd.DataFrame, cols: List[str], skew_threshold: float = 1.0) -> List[str]:
    """
    Seleciona colunas com assimetria (skewness) acima do limiar para aplicar log1p,
    apenas entre as colunas fornecidas e ignorando valores negativos.
    """
    skewed = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if (s > 0).sum() == 0:
            continue
        sk = s.skew(skipna=True)
        if pd.notna(sk) and sk >= skew_threshold:
            skewed.append(c)
    return skewed


def _apply_log1p(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Aplica log1p apenas às colunas indicadas, com proteção de domínio (clip>=0).
    """
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=0)
        out[c] = np.log1p(out[c])
    return out


def _remove_constant_variance(df: pd.DataFrame, cols: List[str], eps: float = 1e-12) -> Tuple[List[str], List[str]]:
    """
    Remove colunas com variância ~0 dentro de 'cols'. Retorna (cols_filtradas, removidas).
    """
    if not cols:
        return [], []
    variances = df[cols].var(numeric_only=True)
    const_cols = variances[variances <= eps].index.tolist()
    kept = [c for c in cols if c not in const_cols]
    return kept, const_cols


def _sanitize_infinite(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Substitui inf/-inf/NaN por 0.0 nas colunas indicadas (barato e seguro antes do scaler)."""
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# =========================================================
# API principal
# =========================================================
def standardize_for_clustering(
    df: pd.DataFrame,
    *,
    cluster_var: List[str],
    log1p_auto: bool = True,
    skew_threshold: float = 1.0,
    fillna_value: float = 0.0,
    drop_constant: bool = True,
    scaler: Optional[StandardScaler] = None,   # se fornecido → modo inferência (transform only)
    used_cols: Optional[List[str]] = None,     # obrigatório quando scaler é fornecido
    return_matrix: bool = False,               # retorna também np.ndarray (Xz) além do DataFrame
) -> Tuple[pd.DataFrame, StandardScaler, List[str], Dict[str, Any]]:
    """
    Padroniza features para clustering com pipeline:
      (seleção) → (log1p opcional por skew) → fillna(0) → (remover var. ~0) → z-score.

    Modo treino:
      - scaler=None → fit_transform nas colunas presentes em 'cluster_var' (após filtros).
      - Retorna df_std com sufixo "_z" e o scaler treinado.

    Modo inferência:
      - scaler!=None e 'used_cols' informado → apenas transform nas colunas 'used_cols' (mesma ordem do treino).
      - Útil para produção/predição reprodutível.

    Retornos:
      - df_std: DataFrame com colunas *_z
      - scaler: StandardScaler (o mesmo recebido em inferência, ou o treinado)
      - used_cols: lista final de colunas usadas para z-score (ordem imutável)
      - meta: dict com detalhes (missing, log1p_cols, removed_constant_cols, skew_threshold, drop_constant)
    """
    if not isinstance(cluster_var, list) or not cluster_var:
        raise ValueError("Parâmetro 'cluster_var' deve ser uma lista não vazia de nomes de colunas.")

    X = df.copy()

    # 1) Seleção de colunas (modo treino usa cluster_var; modo inferência usa used_cols)
    if scaler is None:
        present, missing = _select_present_columns(X, cluster_var)
        if not present:
            raise ValueError(f"Nenhuma coluna de 'cluster_var' encontrada no DataFrame. cluster_var={cluster_var}")
        if missing:
            logger.warning("Colunas ausentes em cluster_var serão ignoradas: %s", missing)
        cols_for_pipeline = present
    else:
        # inferência: precisamos das MESMAS colunas e ORDEM usadas no treino
        if not used_cols:
            raise ValueError("Em modo inferência, forneça 'used_cols' (lista na mesma ordem do treino).")
        cols_for_pipeline = used_cols
        missing_at_inference = [c for c in cols_for_pipeline if c not in X.columns]
        if missing_at_inference:
            raise ValueError(f"Colunas esperadas na inferência não encontradas: {missing_at_inference}")
        missing = []  # apenas informativo para meta

    # 2) log1p automático por assimetria (somente no treino)
    log1p_cols: List[str] = []
    if scaler is None and log1p_auto:
        log1p_cols = _auto_log1p_columns(X, cols_for_pipeline, skew_threshold=skew_threshold)
        if log1p_cols:
            logger.info("Aplicando log1p nas colunas (skew >= %.2f): %s", skew_threshold, log1p_cols)
            X = _apply_log1p(X, log1p_cols)

    # 3) fillna (apenas nas colunas do pipeline) + sanitização (inf/NaN → 0)
    for c in cols_for_pipeline:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[cols_for_pipeline] = X[cols_for_pipeline].fillna(fillna_value)
    X = _sanitize_infinite(X, cols_for_pipeline)

    # 4) descartar colunas de variância ~0 (opcional, apenas no treino)
    removed_constant_cols: List[str] = []
    if scaler is None and drop_constant:
        kept, removed_constant_cols = _remove_constant_variance(X, cols_for_pipeline)
        if removed_constant_cols:
            logger.warning("Removendo colunas com variância ~0: %s", removed_constant_cols)
        cols_for_pipeline = kept
        if not cols_for_pipeline:
            raise ValueError("Todas as colunas foram removidas por variância ~0.")

    # 5) padronização (z-score)
    #    - treino: fit em StandardScaler; inferência: usar scaler fornecido
    if scaler is None:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X[cols_for_pipeline].values)
    else:
        Z = scaler.transform(X[cols_for_pipeline].values)

    # 6) montar DataFrame padronizado com sufixo _z
    z_cols = [f"{c}_z" for c in cols_for_pipeline]
    df_std = df.copy()
    df_std[z_cols] = Z

    # 7) meta-informações úteis para auditoria/reuso
    meta: Dict[str, Any] = {
        "missing_in_input": missing,
        "log1p_cols": log1p_cols,
        "removed_constant_cols": removed_constant_cols,
        "skew_threshold": skew_threshold,
        "drop_constant": drop_constant,
        "used_cols": cols_for_pipeline,
    }

    if return_matrix:
        # retorna também o array Z para consumo direto em modelos
        return df_std, scaler, cols_for_pipeline, {**meta, "matrix": Z}

    return df_std, scaler, cols_for_pipeline, meta


# =========================================================
# I/O helpers
# =========================================================
def save_standardized_features(
    df_std: pd.DataFrame,
    output_path: Path | str,
    *,
    compression: str = "snappy",
) -> Path:
    """
    Salva o DataFrame padronizado em Parquet com escrita segura (tmp->rename).
    """
    output_path = Path(output_path)
    return save_single_parquet(df_std, output_path, compression=compression)


def save_scaler(
    scaler: StandardScaler,
    output_path: Path | str,
) -> Path:
    """
    Salva o StandardScaler em disco (joblib). Retorna o Path final.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, output_path)
    logger.info("Scaler salvo em: %s", output_path)
    return output_path


def read_customer_features(
    input_path: Path | str,
    *,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lê o Parquet de features do cliente. Pode ler apenas um subconjunto de colunas.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de features não encontrado: {input_path}")
    df = pd.read_parquet(input_path, columns=columns, engine="pyarrow")
    logger.info("Features carregadas de %s (%d linhas, %d colunas)", input_path, df.shape[0], df.shape[1])
    return df
