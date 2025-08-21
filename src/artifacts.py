# src/artifacts.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.io_saver import ensure_dir, save_parquet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from typing import Union

def _first_existing(*paths: Union[str, Path]) -> Path:
    """Retorna o primeiro caminho que existir; caso nenhum exista, devolve o último (para boa mensagem de erro)."""
    last = None
    for p in paths:
        p = Path(p)
        last = p
        if p.exists():
            return p
    return last
# -------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------
def _safe_read_parquet(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    return pd.read_parquet(p)


def _try_read(path: Path | str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as e:
        logger.warning("Falha ao ler %s: %s", p, e)
        return None


def _today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")


# -------------------------------------------------------------------
# 1) cluster_assignments.csv
# -------------------------------------------------------------------
def build_cluster_assignments_csv(
    assignments_named_path: Path | str,
    out_csv_path: Path | str,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Lê assignments com nomes (cluster_assignments_named.parquet),
    adiciona metadados em colunas e salva CSV.
    Retorna o DF final para conferência.
    Espera colunas: ['fk_contact','cluster_id','cluster_name'].
    """
    df = _safe_read_parquet(assignments_named_path)

    if "fk_contact" not in df or "cluster_id" not in df:
        raise ValueError("Esperado 'fk_contact' e 'cluster_id' em assignments.")

    # cluster_name pode estar ausente se ainda não houver rótulos
    if "cluster_name" not in df:
        df["cluster_name"] = pd.NA

    # garante tipos simples
    df["cluster_id"] = df["cluster_id"].astype("Int64")
    df["fk_contact"] = df["fk_contact"].astype("string")

    # metadados em colunas (úteis p/ auditoria)
    meta = extra_metadata or {}
    df["artifact_build_date"] = meta.get("artifact_build_date", _today_str())
    df["model_version"] = meta.get("model_version", "v1")
    df["algorithm"] = meta.get("algorithm", pd.NA)
    df["k"] = meta.get("k", pd.NA)

    out_csv_path = Path(out_csv_path)
    ensure_dir(out_csv_path.parent)
    df.to_csv(out_csv_path, index=False, encoding="utf-8")
    logger.info("cluster_assignments.csv salvo em: %s", out_csv_path)
    return df


# -------------------------------------------------------------------
# 2) cluster_centroids.json
# -------------------------------------------------------------------
def build_cluster_centroids_json(
    centroids_z_path: Path | str,
    centroids_unstd_path: Path | str,
    scaler_path: Path | str,
    used_cols: List[str],
    metrics_path: Optional[Path | str] = None,
    model_info: Optional[Dict[str, str | int | float]] = None,
    out_json_path: Path | str = None,
) -> Dict:
    cz = _safe_read_parquet(centroids_z_path)
    # tenta ler o unstd; se não existir ou estiver sem as colunas esperadas, iremos reconstruir
    need_unstd_cols = set(used_cols)
    try:
        cu = _safe_read_parquet(centroids_unstd_path)
    except FileNotFoundError:
        cu = None

    # carrega scaler para (i) médias globais e (ii) inversão
    scaler = joblib.load(scaler_path)
    if not hasattr(scaler, "mean_"):
        raise TypeError("Scaler não possui atributo 'mean_'. Esperado StandardScaler.")

    # helper: pega matriz z nas colunas certas, aceitando sufixo _z
    def _get_z_matrix(df_z: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        cols_in = []
        for c in cols:
            cz_name = f"{c}_z"
            if cz_name in df_z.columns:
                cols_in.append(cz_name)
            elif c in df_z.columns:
                cols_in.append(c)
            else:
                raise ValueError(f"Coluna de z-score ausente em centróides z: '{c}'/'{c}_z'")
        return df_z[cols_in].copy()

    # Se 'cu' não tem as colunas originais, reconstrói a partir de 'cz' usando o scaler
    rebuild_unstd = False
    if cu is None:
        rebuild_unstd = True
    else:
        missing_unstd = [c for c in used_cols if c not in cu.columns]
        if missing_unstd:
            rebuild_unstd = True

    if rebuild_unstd:
        Z = _get_z_matrix(cz, used_cols).to_numpy(dtype=float)
        # inverse_transform de z -> espaço original
        X_unstd = scaler.inverse_transform(Z)
        cu = pd.DataFrame(X_unstd, columns=used_cols, index=cz.index).copy()
        # garanta cluster_id
        if "cluster_id" in cz.columns:
            cu.insert(0, "cluster_id", cz["cluster_id"].astype(int).to_numpy())
        else:
            # se cluster_id não existir, cria um sequencial
            cu.insert(0, "cluster_id", np.arange(len(cu), dtype=int))
        # opcional: persiste o arquivo reconstruído para futuras execuções
        try:
            outp = Path(centroids_unstd_path)
            ensure_dir(outp.parent)
            cu.to_parquet(outp, index=False)
            logger.info("Reconstruí 'centroids_unstd' por inverse_transform e salvei em: %s", outp)
        except Exception as e:
            logger.warning("Falha ao salvar centróides reconstruídos (unstd): %s", e)

    # até aqui: cz = centróides em z, cu = centróides no espaço original (reais ou reconstruídos)
    global_means = {c: float(m) for c, m in zip(used_cols, scaler.mean_)}

    # métricas do modelo
    model_meta = {"model_version": "v1", "algorithm": "KMeans", "k": int(cz["cluster_id"].nunique())}
    if metrics_path and Path(metrics_path).exists():
        try:
            met = pd.read_parquet(metrics_path)
            # espera colunas tipo: ['k','algorithm','inertia','silhouette','is_final']
            fin = met[met.get("is_final", False) == True]
            if fin.empty:
                fin = met.sort_values(by="silhouette", ascending=False).head(1)
            for fld in ["algorithm", "k", "inertia", "silhouette"]:
                if fld in fin:
                    model_meta[fld] = fin.iloc[0][fld]
        except Exception as e:
            logger.warning("Não foi possível ler metrics_path (%s): %s", metrics_path, e)
    if model_info:
        model_meta.update(model_info)

    # arruma colunas esperadas
    def _records(df: pd.DataFrame, suffix: str) -> List[Dict]:
        cols = {}
        for c in used_cols:
            cz_name = f"{c}_z"
            # para 'z': aceita <col>_z ou <col>; para 'unstd': aceita <col> puro
            if suffix == "z":
                cols[cz_name if cz_name in df.columns else c] = c
            else:
                cols[c] = c  # espaço original: esperamos o nome puro
        base_cols = ["cluster_id"] + list(cols.keys())
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas faltantes em centróides ({suffix}): {missing}")
        recs = []
        for _, row in df[base_cols].iterrows():
            r = {"cluster_id": int(row["cluster_id"])}
            for old, new in cols.items():
                val = row[old]
                r[new] = None if pd.isna(val) else float(val)
            recs.append(r)
        return recs

    centroids_obj = {
        "z": _records(cz, "z"),
        "unstd": _records(cu, "unstd"),
    }

    obj = {
        "model": model_meta,
        "build_date": _today_str(),
        "features": used_cols,
        "centroids": centroids_obj,
        "statistics": {"global_means": global_means},
    }

    if out_json_path:
        out_json_path = Path(out_json_path)
        ensure_dir(out_json_path.parent)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info("cluster_centroids.json salvo em: %s", out_json_path)

    return obj


# -------------------------------------------------------------------
# 3) customer_features.parquet (recarimbar / garantir output)
# -------------------------------------------------------------------
def copy_customer_features(
    customer_features_path: Path | str,
    out_parquet_path: Path | str,
    extra_metadata_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Lê as features finais (já geradas na etapa 4/5) e salva novamente no diretório de outputs,
    adicionando (opcionalmente) algumas colunas de metadados.
    """
    df = _safe_read_parquet(customer_features_path).copy()
    if extra_metadata_cols:
        for k, v in extra_metadata_cols.items():
            df[k] = v
    save_parquet(df, out_parquet_path)
    logger.info("customer_features.parquet salvo/copiado em: %s", out_parquet_path)
    return df


# -------------------------------------------------------------------
# 4) cluster_summary.xlsx
#     - overview: tamanho, médias por variável (usadas no cluster)
#     - top rotas: ranking por cluster a partir da feature 'top_route_out'
#     - top viações: ranking por cluster a partir da feature 'top_company_out'
#     - glossário: explicações breves de colunas e artefatos
# -------------------------------------------------------------------
def _rank_top(
    df_feats: pd.DataFrame,
    assignments: pd.DataFrame,
    key_col: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Gera um ranking por cluster da coluna categórica `key_col` (ex.: 'top_route_out').
    Retorna colunas: cluster_id, key_col, count, share (por cluster).
    """
    if key_col not in df_feats.columns:
        # se a coluna não existir nas features, retorna vazio (mas com colunas)
        return pd.DataFrame(columns=["cluster_id", key_col, "count", "share"])

    # junta cluster_id por cliente
    cols_need = ["fk_contact", "cluster_id"]
    ass = assignments[cols_need].dropna().copy()
    ass["cluster_id"] = ass["cluster_id"].astype(int)

    base = df_feats[["fk_contact", key_col]].merge(ass, on="fk_contact", how="inner")
    base[key_col] = base[key_col].astype("string").fillna("NA")

    # contagem por cluster x valor
    ct = base.groupby(["cluster_id", key_col]).size().rename("count").reset_index()

    # normaliza (share por cluster)
    sizes = ct.groupby("cluster_id")["count"].sum().rename("tot").reset_index()
    out = ct.merge(sizes, on="cluster_id", how="left")
    out["share"] = np.where(out["tot"] > 0, out["count"] / out["tot"], 0.0)
    out = out.drop(columns="tot")

    # top N por cluster
    out = out.sort_values(["cluster_id", "count"], ascending=[True, False])
    out = out.groupby("cluster_id").head(top_n).reset_index(drop=True)
    return out


def build_cluster_summary_xlsx(
    assignments_named_path: Path | str,
    centroid_summary_path: Path | str,
    customer_features_path: Path | str,
    used_cols: List[str],
    out_xlsx_path: Path | str,
) -> None:
    """
    Cria um Excel com 4 abas: overview, top_rotas, top_viacoes, glossario.
    - overview: tamanho dos clusters e médias por variável usada.
    - top_rotas: ranking por 'top_route_out' nas features.
    - top_viacoes: ranking por 'top_company_out' nas features.
    - glossario: notas rápidas sobre variáveis e artefatos.
    """
    ass = _safe_read_parquet(assignments_named_path)
    cs = _safe_read_parquet(centroid_summary_path)  # z + unstd + labels
    feats = _safe_read_parquet(customer_features_path)

    # ---------------- overview ----------------
    # tamanho
    size = ass.groupby("cluster_id").size().rename("n_customers").reset_index()
    size["cluster_id"] = size["cluster_id"].astype(int)

    # médias (no espaço original das features usadas)
    for c in used_cols:
        if c not in feats.columns:
            feats[c] = np.nan
    means = feats.merge(ass[["fk_contact", "cluster_id"]], on="fk_contact", how="inner") \
                 .groupby("cluster_id")[used_cols].mean(numeric_only=True).reset_index()

    # junta com centróides (útil p/ confronto)
    # Observação: 'cs' contém colunas <col>_z e/ou <col> (unstd), além de *_label
    # Vamos extrair valores unstd, se existirem, com o mesmo nome da coluna original.
    cols_unstd = [c for c in used_cols if c in cs.columns]
    view_cs = cs[["cluster_id"] + cols_unstd].copy() if cols_unstd else cs[["cluster_id"]].copy()

    overview = size.merge(means, on="cluster_id", how="left").merge(view_cs, on="cluster_id", how="left")
    # renomeia colunas duplicadas (média vs centróide unstd)
    rename_map = {}
    for c in used_cols:
        if c in view_cs.columns:
            rename_map[c] = f"{c}_centroid_unstd"
    overview = overview.rename(columns=rename_map)

    # ---------------- tops ----------------
    top_rotas = _rank_top(feats, ass, key_col="top_route_out", top_n=15)
    top_viacoes = _rank_top(feats, ass, key_col="top_company_out", top_n=15)

    # ---------------- glossario ----------------
    gloss = pd.DataFrame({
        "campo": [
            "fk_contact", "cluster_id", "cluster_name",
            *used_cols,
            "top_route_out", "top_company_out",
            "artifact_build_date", "model_version", "algorithm", "k"
        ],
        "descricao": [
            "Identificador do cliente",
            "ID do cluster",
            "Nome legível do cluster (se rotulado)",
            *[
                "Variável usada no clustering (espaço original)" for _ in used_cols
            ],
            "Rota mais usada pelo cliente (ida)",
            "Viação mais usada pelo cliente (ida)",
            "Data de geração do artefato",
            "Versão do modelo",
            "Algoritmo de clusterização",
            "Número de clusters (k)"
        ]
    })

    # ---------------- gravar Excel ----------------
    out_xlsx_path = Path(out_xlsx_path)
    ensure_dir(out_xlsx_path.parent)

    # escolhe engine disponível
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            engine = None

    with pd.ExcelWriter(out_xlsx_path, engine=engine) as writer:
        overview.to_excel(writer, index=False, sheet_name="overview")
        top_rotas.to_excel(writer, index=False, sheet_name="top_rotas")
        top_viacoes.to_excel(writer, index=False, sheet_name="top_viacoes")
        gloss.to_excel(writer, index=False, sheet_name="glossario")

    logger.info("cluster_summary.xlsx salvo em: %s", out_xlsx_path)


# -------------------------------------------------------------------
# 5) Orquestrador geral desta etapa
# -------------------------------------------------------------------
def run_artifacts_pipeline(
    # entradas
    outputs_dir: Path | str,
    clustering_dir: Path | str,
    models_dir: Path | str,
    used_cols: List[str],
    # nomes padrão
    assignments_named_filename: str = "cluster_assignments_named.parquet",
    centroids_z_filename: str = "cluster_centroids_z.parquet",
    centroids_unstd_filename: str = "cluster_centroids_unstd.parquet",
    centroid_summary_filename: str = "centroid_summary.parquet",
    scaler_filename: str = "standard_scaler.joblib",
    metrics_filename: Optional[str] = "cluster_eval_metrics.parquet",
    customer_features_filename: str = "customer_features.parquet",
    # saídas
    out_assign_csv: str = "cluster_assignments.csv",
    out_centroids_json: str = "cluster_centroids.json",
    out_customer_feats: str = "customer_features.parquet",
    out_summary_xlsx: str = "cluster_summary.xlsx",
    # meta
    model_info: Optional[Dict[str, str | int | float]] = None,
) -> Dict[str, Path]:
    """
    Gera todos os artefatos solicitados e devolve os caminhos dos arquivos criados.
    """
    outputs_dir = Path(outputs_dir)
    clustering_dir = Path(clustering_dir)
    models_dir = Path(models_dir)

    # paths de entrada
    p_assign_named = _first_existing(
        clustering_dir / "labels" / assignments_named_filename,   # onde está no seu caso
        clustering_dir / assignments_named_filename               # fallback na raiz
    )
    p_centroids_z = clustering_dir / centroids_z_filename
    p_centroids_unstd = clustering_dir / centroids_unstd_filename
    p_centroid_summary = _first_existing(
        clustering_dir / "labels" / centroid_summary_filename,    # normalmente salvo aqui pela etapa 7
        clustering_dir / centroid_summary_filename                # fallback
    )
    p_scaler = models_dir / scaler_filename
    p_metrics = None if metrics_filename is None else (clustering_dir / metrics_filename)
    p_customer_feats_in = outputs_dir / customer_features_filename  # já existe da etapa 4/5

    # diretórios de saída
    ensure_dir(outputs_dir)
    ensure_dir(clustering_dir / "labels")

    # 1) assignments.csv
    out_assign_csv_path = outputs_dir / out_assign_csv
    assign_meta = {"artifact_build_date": _today_str()}
    if model_info:
        assign_meta.update(model_info)
    _ = build_cluster_assignments_csv(
        assignments_named_path=p_assign_named,
        out_csv_path=out_assign_csv_path,
        extra_metadata=assign_meta,
    )

    # 2) cluster_centroids.json
    out_centroids_json_path = outputs_dir / out_centroids_json
    _ = build_cluster_centroids_json(
        centroids_z_path=p_centroids_z,
        centroids_unstd_path=p_centroids_unstd,
        scaler_path=p_scaler,
        used_cols=used_cols,
        metrics_path=p_metrics if p_metrics and p_metrics.exists() else None,
        model_info=model_info,
        out_json_path=out_centroids_json_path,
    )

    # 3) customer_features.parquet (recarimbar para outputs)
    out_customer_feats_path = outputs_dir / out_customer_feats
    _ = copy_customer_features(
        customer_features_path=p_customer_feats_in,
        out_parquet_path=out_customer_feats_path,
        extra_metadata_cols={"artifact_build_date": _today_str()},
    )

    # 4) cluster_summary.xlsx
    out_summary_xlsx_path = outputs_dir / out_summary_xlsx
    build_cluster_summary_xlsx(
        assignments_named_path=p_assign_named,
        centroid_summary_path=p_centroid_summary,
        customer_features_path=p_customer_feats_in,
        used_cols=used_cols,
        out_xlsx_path=out_summary_xlsx_path,
    )

    return {
        "cluster_assignments_csv": out_assign_csv_path,
        "cluster_centroids_json": out_centroids_json_path,
        "customer_features_parquet": out_customer_feats_path,
        "cluster_summary_xlsx": out_summary_xlsx_path,
    }
