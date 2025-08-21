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
    cluster_labeler,   # << novo
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
    """
    Se config.CLUSTER_VARS não existir ou estiver vazio, inferimos automaticamente:
    - somente colunas numéricas
    - exclui a chave do contato (fk_contact ou o que estiver em config.CONTACT_ID)
    - exclui colunas obviamente não-modeláveis (ex.: datas)
    """
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

        # 2) Rodar pipeline de limpeza
        results = cleaning.run_cleaning_pipeline(df_raw, tz=config.TIMEZONE)

        # 3) Salvar resultados intermediários
        output_path = config.RUN_OUTPUT_PATH
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                out_file = output_path / f"{name}.parquet"
                io_saver.save_single_parquet(df, out_file)

                # Para df_clean: salvar particionado e seguir com janela + features
                if name == "df_clean":
                    io_saver.save_partitioned_by_year(
                        df,
                        root_dir=output_path,
                        dataset_name="df_clean_partitioned",
                        partition_cols=["year", "month"],
                        overwrite=True,
                    )

                    # 4) Construir e salvar janela de observação
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

                    # 5) Construir features de cliente e salvar
                    df_feats = feature_builder.build_customer_features(
                        df_window,
                        build_date=config.DEFAULT_BUILD_DATE,
                        model_version=config.MODEL_VERSION,
                    )
                    feat_path = config.path_customer_features(output_path)
                    feature_builder.save_customer_features(df_feats, out_path=feat_path)

                    # ===========================
                    # 6) Padronização (standardizer.py)
                    # ===========================
                    try:
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
                    except Exception as e_std:
                        logger.exception("Falha na etapa de padronização: %s", e_std)
                        raise

                    # ===========================
                    # 7) Clustering (cluster_trainer.py)
                    # ===========================
                    try:
                        # Matrizes e parâmetros
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

                        # 7.1 Avaliação de k
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
                        k_best = cluster_trainer._choose_k(evals)  # ok usar aqui
                        logger.info("k* escolhido: %d", k_best)

                        # 7.2 Treino final
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

                        # 7.5 Métricas de avaliação (DataFrame)
                        metrics_df = pd.DataFrame(
                            [{"k": e.k, "inertia": e.inertia, "silhouette": e.silhouette} for e in evals]
                        ).assign(
                            algorithm=algorithm,
                            random_state=random_state,
                            is_final=lambda d: d["k"] == k_best
                        )

                        # 7.6 Montar artifacts e salvar
                        artifacts = {
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
                            },
                        }

                        cluster_outdir = output_path / "clustering"
                        paths = cluster_trainer.save_clustering_artifacts(artifacts, cluster_outdir)
                        logger.info("Artefatos de clustering salvos em: %s", cluster_outdir)
                        for k, p in paths.items():
                            logger.info(" - %s: %s", k, p)

                        # ===========================
                        # 8) Rotulagem e nomes (cluster_labeler.py)
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
                            meta=artifacts.get("meta", None),
                        )

                        labels_dir = cluster_outdir / "labels"
                        label_paths = cluster_labeler.save_labeling_outputs(
                            label_outputs,
                            labels_dir,
                            save_csv_extras=True,
                        )
                        logger.info("Artefatos de rotulagem salvos em: %s", labels_dir)
                        for k, p in label_paths.items():
                            logger.info(" - %s: %s", k, p)

                    except Exception as e_clu:
                        logger.exception("Falha na etapa de clustering/rotulagem: %s", e_clu)
                        raise

        logger.info("Pipeline concluído com sucesso!")

    except Exception as e:
        logger.exception("Falha ao executar pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
