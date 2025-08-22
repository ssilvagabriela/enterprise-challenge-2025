# app.py — Dashboard interativo do pipeline (Streamlit)
# ---------------------------------------------------
# Pré-requisitos:
#   pip install streamlit pandas plotly joblib openpyxl
#   (e as mesmas libs usadas no seu projeto: scikit-learn, etc.)
# Execução:
#   streamlit run app.py

from __future__ import annotations
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

# Importa os módulos do seu projeto
from src import (
    data_loader,
    cleaning,
    io_saver,
    windowing,
    feature_builder,
    standardizer,
    cluster_trainer,
    cluster_labeler,
    artifacts,
    config,
)

# ----------------------------
# Utilidades de cache/estado
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_csv_cached(path: str) -> pd.DataFrame:
    return data_loader.load_purchases(path)

@st.cache_data(show_spinner=False)
def _save_parquet(df: pd.DataFrame, path: Path) -> str:
    io_saver.save_single_parquet(df, path)
    return str(path)

# Para objetos "pesados" como scaler/model
@st.cache_resource(show_spinner=False)
def _cache_resource(obj):
    return obj

# ----------------------------
# Sidebar — parâmetros do experimento
# ----------------------------
st.set_page_config(page_title="NexTrip AI — Pipeline ClickBus", layout="wide")
st.title("NexTrip AI · Dashboard do Pipeline de Segmentação")

with st.sidebar:
    st.header("Parâmetros")

    # Entrada de dados
    st.subheader("Dados de entrada")
    data_source = st.radio(
        "Como quer fornecer o dataset?",
        ["Arquivo CSV local", "Caminho (string)"],
        index=0,
    )
    csv_bytes = None
    csv_path = None
    if data_source == "Arquivo CSV local":
        up = st.file_uploader("Selecione df_t.csv (ou similar)", type=["csv"])
        if up is not None:
            csv_bytes = up.getvalue()
    else:
        csv_path = st.text_input("Caminho para CSV", value="df_t.csv")

    # Diretório base de saída (cada execução cria um subpasta com timestamp)
    base_out = st.text_input(
        "Diretório base de saída",
        value=str(getattr(config, "RUN_OUTPUT_PATH", Path("runs")).absolute()),
    )

    st.subheader("Janela e construção de features")
    build_date = st.date_input("Build date (DEFAULT_BUILD_DATE)", value=getattr(config, "DEFAULT_BUILD_DATE", date.today()))
    janela_meses = st.number_input("Janela de observação (meses)", min_value=1, max_value=36, value=12)
    min_rows = st.number_input("Mínimo de linhas pós-filtro", min_value=0, value=1000, step=100)

    st.subheader("Padronização")
    log1p_auto = st.checkbox("Aplicar log1p automático", value=True)
    skew_thr = st.number_input("Limiar de skew para log1p", min_value=0.0, value=1.0, step=0.1)
    fillna_val = st.number_input("Preencher NaN com", value=0.0, step=0.1)
    drop_constant = st.checkbox("Remover colunas ~constantes", value=True)
    z_suffix = st.text_input("Sufixo de z-score", value=getattr(config, "Z_SUFFIX", "_z"))

    st.subheader("Clustering")
    algorithm = st.selectbox("Algoritmo", options=["kmeans", "minibatch_kmeans"], index=0)
    k_values_str = st.text_input("Valores de k (ex.: 2,3,4,5,6,7,8,9,10)", value=",".join(map(str, getattr(config, "CLUSTER_K_VALUES", list(range(2, 11))))))
    random_state = st.number_input("random_state", value=getattr(config, "RANDOM_STATE", 42))
    n_init = st.number_input("k-means n_init", min_value=1, value=getattr(config, "KMEANS_N_INIT", 10))
    max_iter = st.number_input("k-means max_iter", min_value=50, value=getattr(config, "KMEANS_MAX_ITER", 300))
    batch_size = st.number_input("MiniBatch batch_size", min_value=64, value=getattr(config, "MBKMEANS_BATCH_SIZE", 1024), step=64)
    silhouette_max_samples = st.number_input("Amostras para silhouette (cálculo)", min_value=1000, value=getattr(config, "SILHOUETTE_MAX_SAMPLES", 10000), step=1000)

    st.subheader("Rotulagem")
    low_thr = st.number_input("Threshold baixo (z)", value=getattr(config, "LABEL_LOW_THR", -0.5), step=0.1)
    high_thr = st.number_input("Threshold alto (z)", value=getattr(config, "LABEL_HIGH_THR", 0.5), step=0.1)
    max_frag = st.number_input("Máx. fragmentos no rótulo", min_value=1, value=getattr(config, "LABEL_MAX_FRAGMENTS", 3))

    run_button = st.button("▶ Executar pipeline")

# Inicializa storage de execuções
if "runs" not in st.session_state:
    st.session_state.runs: List[Dict[str, Any]] = []

# ----------------------------
# Funções auxiliares (espelham seu main.py)
# ----------------------------
def _infer_cluster_vars(df_feats: pd.DataFrame) -> List[str]:
    contact_id = getattr(config, "CONTACT_ID", "fk_contact")
    exclude = {contact_id}
    exclude |= {c for c in df_feats.columns if c.endswith("_dt") or c.endswith("_date")}
    num_cols = [c for c in df_feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_feats[c])]
    return num_cols

# ----------------------------
# Execução do pipeline
# ----------------------------
if run_button:
    try:
        t0 = time.time()

        # Define diretório dedicado para esta execução
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_out) / f"run_{build_date.strftime('%Y%m%d')}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1) Carregar dados
        if csv_bytes is not None:
            tmp_csv = run_dir / "input.csv"
            tmp_csv.write_bytes(csv_bytes)
            df_raw = _load_csv_cached(str(tmp_csv))
            input_path_used = str(tmp_csv)
        else:
            if not csv_path:
                st.error("Informe um CSV válido.")
                st.stop()
            df_raw = _load_csv_cached(csv_path)
            input_path_used = csv_path

        # 2) Limpeza
        results = cleaning.run_cleaning_pipeline(df_raw, tz=getattr(config, "TIMEZONE", "America/Sao_Paulo"))

        # 3) Persistir intermediários mínimos
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                _save_parquet(df, run_dir / f"{name}.parquet")

        df_clean = results.get("df_clean")
        if df_clean is None or df_clean.empty:
            st.warning("df_clean não gerado ou vazio.")
            st.stop()

        # 4) Janela
        df_window, start_date, build_dt, end_date = windowing.build_observation_window(
            df_clean,
            build_date=pd.to_datetime(build_date).date(),
            janela_meses=int(janela_meses),
            post_filter_valid=True,
            min_rows=int(min_rows),
        )
        windowing.save_window_parquet(
            df_window,
            out_dir=run_dir,
            janela_meses=int(janela_meses),
            filename_prefix="df_clean_window",
        )

        # 5) Features por cliente
        df_feats = feature_builder.build_customer_features(
            df_window,
            build_date=pd.to_datetime(build_date).date(),
            model_version=getattr(config, "MODEL_VERSION", "v0"),
        )
        feat_path = run_dir / "customer_features.parquet"
        feature_builder.save_customer_features(df_feats, out_path=feat_path)

        # 6) Padronização
        cluster_vars = getattr(config, "CLUSTER_VARS", None)
        if not cluster_vars:
            cluster_vars = _infer_cluster_vars(df_feats)
        df_std, scaler, used_cols, meta = standardizer.standardize_for_clustering(
            df_feats,
            cluster_var=cluster_vars,
            log1p_auto=bool(log1p_auto),
            skew_threshold=float(skew_thr),
            fillna_value=float(fillna_val),
            drop_constant=bool(drop_constant),
            return_matrix=False,
        )
        std_path = run_dir / "customer_features_standardized.parquet"
        scaler_path = run_dir / "models" / "scaler.joblib"
        std_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        standardizer.save_standardized_features(df_std, std_path)
        standardizer.save_scaler(scaler, scaler_path)

        # 7) Clustering
        id_col = getattr(config, "CONTACT_ID", "fk_contact")
        z_cols = [f"{c}{z_suffix}" for c in used_cols]
        X = df_std[z_cols].values

        k_values = [int(k.strip()) for k in k_values_str.split(",") if k.strip()]
        evals = cluster_trainer.evaluate_k_range(
            X,
            k_values=k_values,
            algorithm=algorithm,
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=int(max_iter),
            batch_size=int(batch_size),
            silhouette_max_samples=int(silhouette_max_samples),
        )
        k_best = cluster_trainer._choose_k(evals)
        model, labels = cluster_trainer.fit_final_model(
            X,
            k=k_best,
            algorithm=algorithm,
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=int(max_iter),
            batch_size=int(batch_size),
        )

        assignments = df_std[[id_col]].copy()
        assignments["cluster_id"] = labels.astype(int)
        cent_z, cent_unstd = cluster_trainer.centroids_com_scaler(model, scaler=scaler, used_cols=used_cols)
        metrics_df = pd.DataFrame([{ "k": e.k, "inertia": e.inertia, "silhouette": e.silhouette } for e in evals])
        metrics_df = metrics_df.assign(algorithm=algorithm, is_final=lambda d: d["k"]==k_best)

        cluster_outdir = run_dir / "clustering"
        cluster_outdir.mkdir(parents=True, exist_ok=True)
        cz_path = cluster_outdir / "centroids_z.parquet"
        cu_path = cluster_outdir / "centroids_unstd.parquet"
        mt_path = cluster_outdir / "metrics.parquet"
        asg_path = cluster_outdir / "assignments.parquet"
        io_saver.save_single_parquet(cent_z, cz_path)
        io_saver.save_single_parquet(cent_unstd, cu_path)
        io_saver.save_single_parquet(metrics_df, mt_path)
        io_saver.save_single_parquet(assignments, asg_path)

        artifacts_dict = {
            "assignments": assignments,
            "centroids_z": cent_z,
            "centroids_unstd": cent_unstd,
            "evals": metrics_df,
            "model": model,
            "k_best": k_best,
            "meta": {
                "algorithm": algorithm,
                "random_state": int(random_state),
                "k_values": list(k_values),
                "k_best": int(k_best),
                "used_cols": list(used_cols),
                "z_suffix": z_suffix,
                "build_date": str(build_date),
                "model_version": str(getattr(config, "MODEL_VERSION", "v0")),
            },
        }

        # Orquestração de artefatos
        orch_out = artifacts.orchestrate_artifacts(
            out_dir=cluster_outdir / "artifacts",
            centroids_z_paths=[cz_path],
            used_cols=used_cols,
            centroids_unstd_paths=[cu_path],
            scaler_path=scaler_path,
            metrics_paths=[mt_path],
            assignments_paths=[asg_path],
            model_name=algorithm,
            meta={"build_date": str(build_date), "model_version": str(getattr(config, "MODEL_VERSION", "v0"))},
            excel_top_n=15,
        )

        # Rotulagem
        label_outputs = cluster_labeler.run_labeling_pipeline(
            centroids_z=cent_z,
            used_cols=used_cols,
            assignments=assignments,
            centroids_unstd=cent_unstd,
            metrics=metrics_df,
            cluster_id_col="cluster_id",
            id_col=id_col,
            low_thr=float(low_thr),
            high_thr=float(high_thr),
            max_fragments=int(max_frag),
            aliases=getattr(config, "LABEL_ALIASES", None),
            priority=getattr(config, "LABEL_PRIORITY", None),
            meta=artifacts_dict.get("meta", None),
        )
        labels_dir = cluster_outdir / "labels"
        label_paths = cluster_labeler.save_labeling_outputs(label_outputs, labels_dir, save_csv_extras=True)

        # Excel enriquecido
        try:
            centroid_labels = label_outputs.get("centroid_labels")
            assignments_named = label_outputs.get("assignments_named")
            rich_xlsx = artifacts.build_cluster_summary_xlsx(
                out_path=labels_dir / "cluster_summary_labeled.xlsx",
                centroids_z=cent_z,
                centroid_labels=centroid_labels if centroid_labels is not None else pd.DataFrame(),
                assignments_named=assignments_named if assignments_named is not None and not assignments_named.empty else None,
                centroids_unstd=cent_unstd,
                metrics=None,
                glossary=None,
                top_n=15,
            )
        except Exception:
            rich_xlsx = None

        # Persistir metadados da execução
        run_meta = {
            "run_dir": str(run_dir),
            "input_path": input_path_used,
            "params": {
                "build_date": str(build_date),
                "janela_meses": int(janela_meses),
                "min_rows": int(min_rows),
                "algorithm": algorithm,
                "k_values": k_values,
                "k_best": int(k_best),
                "log1p_auto": bool(log1p_auto),
                "skew_thr": float(skew_thr),
                "fillna": float(fillna_val),
                "drop_constant": bool(drop_constant),
                "z_suffix": z_suffix,
                "low_thr": float(low_thr),
                "high_thr": float(high_thr),
                "max_frag": int(max_frag),
            },
            "paths": {
                "df_clean": str(run_dir / "df_clean.parquet"),
                "window": str(run_dir / f"df_clean_window_{int(janela_meses)}m.parquet"),
                "features": str(feat_path),
                "standardized": str(std_path),
                "scaler": str(scaler_path),
                "centroids_z": str(cz_path),
                "centroids_unstd": str(cu_path),
                "metrics": str(mt_path),
                "assignments": str(asg_path),
                "orch": {k: str(v) for k, v in orch_out.items()},
                "labels_dir": str(labels_dir),
                "rich_xlsx": str(rich_xlsx) if rich_xlsx else None,
            },
            "timing_s": round(time.time() - t0, 2),
            "ts": ts,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

        st.session_state.runs.append(run_meta)
        st.success("Execução concluída com sucesso!")
    except Exception as e:
        st.exception(e)

# ----------------------------
# Visualização dos resultados (histórico de execuções)
# ----------------------------
st.header("Histórico de execuções")
if not st.session_state.runs:
    st.info("Nenhuma execução ainda. Configure os parâmetros na barra lateral e clique em 'Executar pipeline'.")
else:
    # Tabela com runs
    runs_df = pd.DataFrame([
        {
            "Execução": i + 1,
            "Data": r["params"]["build_date"],
            "k*": r["params"]["k_best"],
            "Algoritmo": r["params"]["algorithm"],
            "Janela (m)": r["params"]["janela_meses"],
            "Tempo (s)": r["timing_s"],
            "run_dir": r["run_dir"],
        }
        for i, r in enumerate(st.session_state.runs)
    ])

    sel = st.dataframe(runs_df, use_container_width=True)
    sel_idx = st.number_input("Selecione o índice da execução para detalhar", min_value=1, max_value=len(st.session_state.runs), value=len(st.session_state.runs))
    run_sel = st.session_state.runs[sel_idx - 1]

    st.subheader("Resumo da execução selecionada")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Build date:**", run_sel["params"]["build_date"])
        st.write("**Janela (meses):**", run_sel["params"]["janela_meses"])
        st.write("**Tempo (s):**", run_sel["timing_s"])    
    with col2:
        st.write("**Algoritmo:**", run_sel["params"]["algorithm"])    
        st.write("**k_values:**", run_sel["params"]["k_values"])    
        st.write("**k*:**", run_sel["params"]["k_best"])    
    with col3:
        st.write("**z_suffix:**", run_sel["params"]["z_suffix"])    
        st.write("**log1p_auto/skew_thr:**", (run_sel["params"]["log1p_auto"], run_sel["params"]["skew_thr"]))
        st.write("**fillna/drop_const:**", (run_sel["params"]["fillna"], run_sel["params"]["drop_constant"]))

    # Carrega métricas e plota
    try:
        metrics_df = pd.read_parquet(run_sel["paths"]["metrics"])
        st.subheader("Métricas por k")
        fig = px.line(metrics_df.sort_values("k"), x="k", y=["inertia", "silhouette"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Métricas não disponíveis.")

    # Tabelas principais
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Centróides (z)**")
        try:
            st.dataframe(pd.read_parquet(run_sel["paths"]["centroids_z"]).set_index("cluster_id"), use_container_width=True, height=360)
        except Exception:
            st.write("—")
    with c2:
        st.markdown("**Centróides (unstd)**")
        try:
            st.dataframe(pd.read_parquet(run_sel["paths"]["centroids_unstd"]).set_index("cluster_id"), use_container_width=True, height=360)
        except Exception:
            st.write("—")

    st.subheader("Atribuições de clientes (amostra)")
    try:
        asg = pd.read_parquet(run_sel["paths"]["assignments"])\
                .sample(min(5000, sum(1 for _ in pd.read_parquet(run_sel["paths"]["assignments"]).itertuples())), random_state=1)
        st.dataframe(asg, use_container_width=True, height=300)
    except Exception:
        try:
            st.dataframe(pd.read_parquet(run_sel["paths"]["assignments"]).head(1000), use_container_width=True, height=300)
        except Exception:
            st.write("—")

    st.subheader("Downloads úteis")
    paths = run_sel["paths"]
    for label, p in [
        ("df_clean.parquet", paths.get("df_clean")),
        ("window.parquet", paths.get("window")),
        ("features.parquet", paths.get("features")),
        ("standardized.parquet", paths.get("standardized")),
        ("metrics.parquet", paths.get("metrics")),
        ("assignments.parquet", paths.get("assignments")),
        ("centroids_z.parquet", paths.get("centroids_z")),
        ("centroids_unstd.parquet", paths.get("centroids_unstd")),
        ("Excel enriquecido (se gerado)", paths.get("rich_xlsx")),
    ]:
        if p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button(label=f"Baixar {label}", data=f, file_name=Path(p).name)

    # Se existir JSON com summary dos clusters gerado pelo orchestrate
    try:
        cc_json = Path(paths.get("orch", {}).get("cluster_centroids_json", ""))
        if cc_json and cc_json.exists():
            st.subheader("Centroid summary (JSON)")
            st.code(cc_json.read_text()[:8000])
    except Exception:
        pass

# ============================
# Visualizações adicionais — clusters
# ============================
st.header("Visualizações dos clusters")

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

if st.session_state.runs:
    run_sel = st.session_state.runs[sel_idx - 1]
    paths = run_sel["paths"]
    z_suffix = run_sel["params"].get("z_suffix", "_z")

    # ---- 1) Tamanho dos clusters (barra)
    try:
        asg_df = pd.read_parquet(paths["assignments"]).copy()
        size_df = asg_df.groupby("cluster_id").size().reset_index(name="n_clientes").sort_values("cluster_id")
        st.subheader("Tamanho dos clusters")
        fig_sz = px.bar(size_df, x="cluster_id", y="n_clientes")
        st.plotly_chart(fig_sz, use_container_width=True)
    except Exception as e:
        st.info(f"Não foi possível montar o gráfico de tamanhos: {e}")

    # ---- 2) Projeção 2D (PCA) dos clientes
    st.subheader("Projeção 2D dos clientes (PCA)")
    try:
        if PCA is None:
            st.warning("scikit-learn não disponível para PCA. Instale 'scikit-learn'.")
        else:
            # Lê features padronizadas e junta com o cluster assign
            df_std = pd.read_parquet(paths["standardized"])  # contém fk_contact + colunas z
            asg_df = pd.read_parquet(paths["assignments"]).copy()
            id_col = getattr(config, "CONTACT_ID", "fk_contact")
            dfm = df_std.merge(asg_df, on=id_col, how="left")
            z_cols = [c for c in dfm.columns if c.endswith(z_suffix)]

            # Amostra para não explodir o navegador
            max_points = 20000
            if len(dfm) > max_points:
                dfm = dfm.sample(max_points, random_state=1)

            pca = PCA(n_components=2, random_state=0)
            coords = pca.fit_transform(dfm[z_cols].values)
            proj = pd.DataFrame(coords, columns=["PC1", "PC2"]).assign(cluster_id=dfm["cluster_id"].values)
            fig_scatter = px.scatter(
                proj, x="PC1", y="PC2", color="cluster_id", opacity=0.7,
                title="PCA dos z-scores por cliente"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Dica: clusters separáveis tendem a formar "
                       "grupos bem definidos neste espaço 2D; sobreposição indica similaridade.")
    except Exception as e:
        st.info(f"Não foi possível montar o scatter PCA: {e}")

    # ---- 3) Radar dos centróides (top variáveis por |z|)
    st.subheader("Perfil dos clusters (radar por centróide)")
try:
    cz = pd.read_parquet(paths["centroids_z"]).copy()

    # --- Robustez de formato: coluna/índice/long format ---
    # Se vier com MultiIndex, resetamos
    if isinstance(cz.index, pd.MultiIndex):
        cz = cz.reset_index()
    # Se 'cluster_id' não é coluna, tente promover índice
    if "cluster_id" not in cz.columns:
        cz = cz.reset_index()
        if "cluster_id" not in cz.columns:
            if "index" in cz.columns:
                cz = cz.rename(columns={"index": "cluster_id"})
            elif "cluster" in cz.columns:
                cz = cz.rename(columns={"cluster": "cluster_id"})
    # Se vier no formato longo (cluster_id, feature, z), pivotamos para wide
    if {"cluster_id", "feature", "z"}.issubset(set(cz.columns)):
        cz = cz.pivot(index="cluster_id", columns="feature", values="z").reset_index()

    # Converte cluster_id para numérico quando possível
    try:
        cz["cluster_id"] = pd.to_numeric(cz["cluster_id"]).astype(int)
    except Exception:
        pass

    # Lista de clusters disponíveis
    cluster_choices = sorted(cz["cluster_id"].unique().tolist())
    clust_opt = st.selectbox("Cluster para inspecionar", cluster_choices, index=0)

    top_n = int(st.slider("Top variáveis por |z|", min_value=3, max_value=20, value=8))
    row = cz.loc[cz["cluster_id"] == clust_opt].iloc[0]

    # Colunas numéricas (z) do centróide
    z_cols_in_cz = [c for c in row.index if c != "cluster_id"]
    s = row[z_cols_in_cz].astype(float)
    vals = s.sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)

    rad = pd.DataFrame({
        "feature": [str(c).replace(z_suffix, "") for c in vals.index],
        "z": vals.values,
    })
    rad = pd.concat([rad, rad.iloc[[0]]])  # fecha o polígono

    fig_radar = px.line_polar(rad, r="z", theta="feature", line_close=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Valores positivos (z) indicam o centróide acima da média global na variável; negativos, abaixo.")
except Exception as e:
    st.info(f"Não foi possível montar o radar dos centróides: {e}")

st.caption("NexTrip AI · Clique em 'Executar pipeline' após definir os parâmetros na barra lateral. Cada execução cria uma pasta dedicada em 'Diretório base de saída' contendo todos os artefatos.")
