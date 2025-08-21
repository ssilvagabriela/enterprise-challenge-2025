from pathlib import Path
import pandas as pd
import traceback
import sys

from src.config import INTERMEDIATE_PATH, OUTPUT_PATH
from src import schema_config as sc
from src.feature_builder import build_customer_features, save_customer_features

def main():
    # ---- parâmetros ----
    BUILD_DATE = "2024-01-01"
    MODEL_VERSION = "v1"
    JANELA_MESES = 12  # usado no nome do parquet da janela

    print("[INFO] Python:", sys.version)
    print("[INFO] INTERMEDIATE_PATH:", Path(INTERMEDIATE_PATH).resolve())
    print("[INFO] OUTPUT_PATH:", Path(OUTPUT_PATH).resolve())

    try:
        # 1) ler parquet da janela
        window_parquet = Path(INTERMEDIATE_PATH) / f"purchases_clean_window_{JANELA_MESES}m.parquet"
        print("[INFO] Lendo janela:", window_parquet.resolve())
        if not window_parquet.exists():
            raise FileNotFoundError(f"Arquivo de janela não encontrado: {window_parquet.resolve()}")

        df_cf = pd.read_parquet(window_parquet)
        print("[INFO] df_cf shape:", df_cf.shape)
        print("[INFO] colunas (até 15):", list(df_cf.columns)[:15])

        # 2) construir features
        feats = build_customer_features(df_cf, build_date=BUILD_DATE, model_version=MODEL_VERSION)
        print("[INFO] feats shape:", feats.shape)
        if sc.CONTACT_ID in feats:
            print("[INFO] contatos únicos:", feats[sc.CONTACT_ID].nunique())

        # 3) salvar
        out_path = Path(OUTPUT_PATH) / "customer_features.parquet"
        print("[INFO] Salvando em:", out_path.resolve())
        save_customer_features(feats, out_path)

        # 4) verificar persistência
        exists = out_path.exists()
        size = out_path.stat().st_size if exists else 0
        print(f"[OK] customer_features salvo | caminho: {out_path.resolve()} | existe? {exists} | tamanho: {size} bytes")

        # dump debug opcional (primeiras linhas)
        debug_csv = Path(OUTPUT_PATH) / "customer_features_head.csv"
        feats.head(1000).to_csv(debug_csv, index=False)
        print(f"[OK] dump debug (head 1000): {debug_csv.resolve()} | existe? {debug_csv.exists()}")

    except Exception as e:
        print("[ERRO] Falha na execução do main_features.py")
        print("       ", repr(e))
        print("---- Traceback ----")
        traceback.print_exc()

if __name__ == "__main__":
    main()
