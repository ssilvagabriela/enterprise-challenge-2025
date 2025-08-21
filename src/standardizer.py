from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import schema_config as sc  # usa sc.CONTACT_ID
from src.io_saver import ensure_dir  # já criamos antes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------
# util: decidir colunas de log
# ---------------------------
def _auto_log_columns(df: pd.DataFrame, cols: List[str], skew_thr: float = 0.75) -> List[str]:
    """
    Escolhe automaticamente colunas para log1p:
    - ignora colunas com valores negativos
    - escolhe as com |skewness| >= skew_thr (assimétricas à direita na prática)
    """
    candidates: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        ser = pd.to_numeric(df[c], errors="coerce")
        # precisa ter pelo menos algum valor não-nulo
        if ser.notna().sum() == 0:
            continue
        # se houver negativos, não aplicar log1p
        if (ser.dropna() < 0).any():
            continue
        skew = ser.skew()
        if abs(skew) >= skew_thr:
            candidates.append(c)
    return candidates


def _apply_log1p(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            # protege contra negativos eventuais
            ser = pd.to_numeric(out[c], errors="coerce")
            ser = ser.clip(lower=0)  # garante domínio de log1p
            out[c] = np.log1p(ser)
    return out


# ---------------------------
# função principal de padronização
# ---------------------------
def standardize_for_clustering(
    df: pd.DataFrame,
    cluster_var: List[str],
    log_cols: Union[str, List[str], None] = "auto",
    return_scaler: bool = True,
) -> Tuple[pd.DataFrame, Optional[StandardScaler], List[str]]:
    """
    Retorna um DataFrame com colunas padronizadas via z-score (StandardScaler),
    preservando a coluna de id (sc.CONTACT_ID). Também retorna o scaler (opcional)
    e a lista final de colunas usadas.

    Parâmetros:
    - df: DataFrame com as features por cliente
    - cluster_var: lista de colunas a usar no clustering
    - log_cols:
        - "auto": escolhe automaticamente por skewness
        - list[str]: aplica log1p exatamente nelas
        - None: não aplica log
    - return_scaler: se True, devolve o objeto StandardScaler treinado

    Saídas:
    - X_std: DataFrame com [sc.CONTACT_ID] + colunas padronizadas (sufixo "_z")
    - scaler: StandardScaler treinado (ou None)
    - used_cols: lista das colunas efetivamente utilizadas
    """
    if sc.CONTACT_ID not in df.columns:
        raise ValueError(f"Coluna de id '{sc.CONTACT_ID}' não encontrada no DataFrame.")

    # 1) seleciona somente as colunas de interesse (presentes)
    present = [c for c in cluster_var if c in df.columns]
    missing = [c for c in cluster_var if c not in df.columns]
    if missing:
        logging.warning(f"Colunas ausentes e ignoradas: {missing}")

    X = df[[sc.CONTACT_ID] + present].copy()

    # garante numérico nas colunas de cluster
    for c in present:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 2) decide colunas para log1p
    if log_cols == "auto":
        log_candidates = _auto_log_columns(X, present, skew_thr=0.75)
        logging.info(f"Colunas com log1p (auto): {log_candidates}")
    elif isinstance(log_cols, list):
        log_candidates = [c for c in log_cols if c in present]
        logging.info(f"Colunas com log1p (forçado): {log_candidates}")
    else:
        log_candidates = []

    # 3) aplica log1p nas escolhidas
    X = _apply_log1p(X, log_candidates)

    # 4) preenche NaN com 0 nas colunas de cluster (não mexe no id)
    X[present] = X[present].fillna(0.0)

    # 5) z-score
    scaler = StandardScaler()
    Z = scaler.fit_transform(X[present].values)
    zcols = [f"{c}_z" for c in present]
    X_std = pd.DataFrame(Z, columns=zcols, index=X.index)
    X_std.insert(0, sc.CONTACT_ID, X[sc.CONTACT_ID].values)

    return (X_std, scaler if return_scaler else None, present)


# ---------------------------
# IO helpers
# ---------------------------
def read_customer_features(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    df = pd.read_parquet(p)
    return df


def save_standardized_features(
    df_std: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)
    df_std.to_parquet(out, index=False)
    logging.info(f"Arquivo salvo em: {out} | linhas={len(df_std)} | cols={df_std.shape[1]}")
    return out


def save_scaler(
    scaler: StandardScaler,
    output_path: Union[str, Path],
) -> Path:
    import joblib
    out = Path(output_path)
    ensure_dir(out.parent)
    joblib.dump(scaler, out)
    logging.info(f"Scaler salvo em: {out}")
    return out