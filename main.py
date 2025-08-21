# main.py
from __future__ import annotations
import logging
import sys
from pathlib import Path

import pandas as pd

from src import (
    data_loader,
    cleaning,
    io_saver,
    windowing,
    feature_builder,
    standardizer,
    cluster_trainer,
    cluster_labeler,
    artifacts,  # << novo
    config,
)

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def _infer_cluster_vars(df_feats: pd.DataFrame) -> list[str]:
    """Inferir colunas numéricas para clustering quando config.CLUSTER_VARS está vazio."""
    contact_id = getattr(config, "CONTACT_ID", "fk_contact")
    exclude = {contact_id}
    exclude |= {c for c in df_feats.columns if c.endswith("_dt") or c.endswith("_date")}
    num_cols = [
        c for c in df_feats.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_feats[c])
    ]
    return num_cols


def main():
    try:
        # 1) Carregar dados brutos
        logger.info("Iniciando pipeline...")
        df_raw = data_loader.load_purchases("df_t.csv")

        # 2) Limpeza
        results = cleaning.run_cleaning_pipeline(df_raw, tz=config.TIMEZONE)

        # 3) Salvar intermediários + seguir com janela/features
        output_path = config.RUN_OUTPUT_PATH
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                out_file = output_path / f"{name}.parquet"
                io_saver.save_single_parquet(df, out_file)

                if name == "df_clean":
                    # 3.1 Particionado
                    io_saver.save_partitioned_by_year(
                        df,
                        root_dir=output_path,
                        dataset_name="df_clean_partitioned",
                        partition_cols=["year", "month"],
                        overwrite=True,
                    )

                    # 4) Janela de observação
                    df_window, start_date, build_date, end_date = windowing.build_observation_window(
                        df,
                        build_date=config.DEFAULT_BUILD_DATE,
                        janela_meses=getattr(config, "WINDOW_MONTHS", 1),
                        post_filter_valid=True,
                        min_rows=1000,
                    )
                    windowing.save_window_parquet(
                        df_window,
                        out_dir=output_path,
                        janela_meses=getattr(config, "WINDOW_MONTHS", 1),
                        filename_prefix="df_clean_window",
                    )

                    # 5) Features por cliente
                    df_feats = feature_builder.build_customer_features(
                        df_window,
                        build_date=config.DEFAULT_BUILD_DATE,
                        model_version=config.MODEL_VERSION,
                    )
                    feat_path = config.path_customer_features(output_path)
                    feature_builder.save_customer_features(df_feats, out_path=feat_path)

                    # ===========================
                    # 6) Padronização
                    # ===========================
                    cluster_vars = getattr(config, "CLUSTER_VARS", None)
                    if not cluster_vars:
                        cluster_vars = _infer_cluster_vars(df_feats)
                        logger.info("Inferindo CLUSTER_VARS: %s", cluster_vars)

                    df_std, scaler, used_cols, meta = standardizer.standardize_for_clustering(
                        df_feats,
                        cluster_var=cluster_vars,
                        log1p_auto=True,
                        skew_threshold=1.0,
                        fillna_value=0.0,
                        drop_constant=True,
                        return_matrix=False,
                    )

                    # salvar padronizados e scaler
                    std_out_path = output_path / "customer_features_standardized.parquet"
                    scaler_out_path = output_path / "models" / "scaler.joblib"
                    std_out_path.parent.mkdir(parents=True, exist_ok=True)
                    scaler_out_path.parent.mkdir(parents=True, exist_ok=True)
                    standardizer.save_standardized_features(df_std, std_out_path)
                    standardizer.save_scaler(scaler, scaler_out_path)
                    logger.info(
                        "Padronização concluída. Colunas usadas (%d): %s | Log1p: %s | Removidas (var~0): %s",
                        len(used_cols), used_cols, meta.get("log1p_cols"), meta.get("removed_constant_cols"),
                    )

                    # ===========================
                    # 7) Clustering
                    # ===========================
                    z_suffix = getattr(config, "Z_SUFFIX", "_z")
                    id_col = getattr(config, "CONTACT_ID", "fk_contact")
                    z_cols = [f"{c}{z_suffix}" for c in used_cols]
                    X = df_std[z_cols].values

                    k_values = getattr(config, "CLUSTER_K_VALUES", list(range(2, 11)))
                    algorithm = getattr(config, "CLUSTER_ALGORITHM", "kmeans")
                    random_state = getattr(config, "RANDOM_STATE", 42)
                    n_init = getattr(config, "KMEANS_N_INIT", 10)
                    max_iter = getattr(config, "KMEANS_MAX_ITER", 300)
                    batch_size = getattr(config, "MBKMEANS_BATCH_SIZE", 1024)
                    silhouette_max_samples = getattr(config, "SILHOUETTE_MAX_SAMPLES", 10000)

                    # 7.1 Avaliar k
                    evals = cluster_trainer.evaluate_k_range(
                        X,
                        k_values=k_values,
                        algorithm=algorithm,
                        random_state=random_state,
                        n_init=n_init,
                        max_iter=max_iter,
                        batch_size=batch_size,
                        silhouette_max_samples=silhouette_max_samples,
                    )
                    k_best = cluster_trainer._choose_k(evals)
                    logger.info("k* escolhido: %d", k_best)

                    # 7.2 Treinar final
                    model, labels = cluster_trainer.fit_final_model(
                        X,
                        k=k_best,
                        algorithm=algorithm,
                        random_state=random_state,
                        n_init=n_init,
                        max_iter=max_iter,
                        batch_size=batch_size,
                    )

                    # 7.3 Assignments
                    assignments = df_std[[id_col]].copy()
                    assignments["cluster_id"] = labels.astype(int)

                    # 7.4 Centróides (z e despadronizado)
                    cent_z, cent_unstd = cluster_trainer.centroids_com_scaler(
                        model, scaler=scaler, used_cols=used_cols
                    )

                    # 7.5 Métricas
                    metrics_df = pd.DataFrame(
                        [{"k": e.k, "inertia": e.inertia, "silhouette": e.silhouette} for e in evals]
                    ).assign(
                        algorithm=algorithm,
                        random_state=random_state,
                        is_final=lambda d: d["k"] == k_best
                    )

                    # 7.6 Persistir artefatos-base em paths conhecidos (para o orchestrator)
                    cluster_outdir = output_path / "clustering"
                    cluster_outdir.mkdir(parents=True, exist_ok=True)
                    cz_path = cluster_outdir / "centroids_z.parquet"
                    cu_path = cluster_outdir / "centroids_unstd.parquet"
                    mt_path = cluster_outdir / "metrics.parquet"
                    asg_path = cluster_outdir / "assignments.parquet"

                    io_saver.save_single_parquet(cent_z, cz_path)
                    io_saver.save_single_parquet(cent_unstd, cu_path)
                    io_saver.save_single_parquet(metrics_df, mt_path)
                    io_saver.save_single_parquet(assignments, asg_path)

                    # (opcional) também salvar via util do trainer, se você já usa:
                    try:
                        artifacts_dict = {
                            "assignments": assignments,
                            "centroids_z": cent_z,
                            "centroids_unstd": cent_unstd,
                            "evals": metrics_df,
                            "model": model,
                            "k_best": k_best,
                            "meta": {
                                "algorithm": algorithm,
                                "random_state": random_state,
                                "k_values": list(k_values),
                                "k_best": k_best,
                                "used_cols": used_cols,
                                "z_suffix": z_suffix,
                                "build_date": str(config.DEFAULT_BUILD_DATE),
                                "model_version": str(config.MODEL_VERSION),
                            },
                        }
                        _paths = cluster_trainer.save_clustering_artifacts(artifacts_dict, cluster_outdir)
                        for k, p in _paths.items():
                            logger.info("Trainer salvou: %s -> %s", k, p)
                    except Exception as e:
                        logger.warning("save_clustering_artifacts falhou (%s); seguimos com paths locais.", e)

                    # ===========================
                    # 8) Orquestração de artefatos (artifacts.py)
                    #     - Gera cluster_centroids.json
                    #     - Gera CSV/Parquet de assignments
                    #     - Gera Excel básico
                    # ===========================
                    orch_out = artifacts.orchestrate_artifacts(
                        out_dir=cluster_outdir / "artifacts",
                        centroids_z_paths=[cz_path],
                        used_cols=used_cols,
                        centroids_unstd_paths=[cu_path],
                        scaler_path=scaler_out_path,
                        metrics_paths=[mt_path],
                        assignments_paths=[asg_path],
                        model_name=algorithm,
                        meta={
                            "build_date": str(config.DEFAULT_BUILD_DATE),
                            "model_version": str(config.MODEL_VERSION),
                        },
                        excel_top_n=15,
                    )
                    for k, p in orch_out.items():
                        logger.info("Artifacts/orchestrate -> %s: %s", k, p)
                    # (o Excel aqui é o mínimo; um resumo rico vai abaixo, após a rotulagem)  # :contentReference[oaicite:1]{index=1}

                    # ===========================
                    # 9) Rotulagem e nomes
                    # ===========================
                    low_thr = getattr(config, "LABEL_LOW_THR", -0.5)
                    high_thr = getattr(config, "LABEL_HIGH_THR", 0.5)
                    max_frag = getattr(config, "LABEL_MAX_FRAGMENTS", 3)

                    label_outputs = cluster_labeler.run_labeling_pipeline(
                        centroids_z=cent_z,
                        used_cols=used_cols,
                        assignments=assignments,
                        centroids_unstd=cent_unstd,
                        metrics=metrics_df,
                        cluster_id_col="cluster_id",
                        id_col=id_col,
                        low_thr=low_thr,
                        high_thr=high_thr,
                        max_fragments=max_frag,
                        aliases=getattr(config, "LABEL_ALIASES", None),
                        priority=getattr(config, "LABEL_PRIORITY", None),
                        meta=artifacts_dict.get("meta", None),
                    )

                    labels_dir = cluster_outdir / "labels"
                    label_paths = cluster_labeler.save_labeling_outputs(
                        label_outputs,
                        labels_dir,
                        save_csv_extras=True,
                    )
                    logger.info("Rotulagem salva em: %s", labels_dir)
                    for k, p in label_paths.items():
                        logger.info(" - %s: %s", k, p)

                    # 9.1 Excel enriquecido com labels (artifacts.py)
                    try:
                        centroid_labels = label_outputs.get("centroid_labels")
                        assignments_named = label_outputs.get("assignments_named")

                        rich_xlsx = artifacts.build_cluster_summary_xlsx(
                            out_path=labels_dir / "cluster_summary_labeled.xlsx",
                            centroids_z=cent_z,
                            centroid_labels=centroid_labels if centroid_labels is not None else pd.DataFrame(),
                            assignments_named=assignments_named if assignments_named is not None and not assignments_named.empty else None,
                            centroids_unstd=cent_unstd,
                            metrics=None,  # <- evita KeyError 'cluster_id' no merge interno
                            glossary=None,
                            top_n=15,
                        )
                        logger.info("Excel enriquecido salvo em: %s", rich_xlsx)
                    except Exception as e:
                        logger.warning("Falha ao montar Excel enriquecido (%s).", e)


        logger.info("Pipeline concluído com sucesso!")

    except Exception as e:
        logger.exception("Falha ao executar pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
