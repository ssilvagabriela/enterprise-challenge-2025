from pathlib import Path

# Caminho base do projeto
BASE_PATH = Path(r"C:\Users\gabis\Downloads\enterprise-challenge-2025")

# Estrutura de pastas
RAW_PATH = BASE_PATH / "data_raw"
INTERMEDIATE_PATH = BASE_PATH / "data_intermediate"
OUTPUT_PATH = BASE_PATH / "data_outputs"

for path in [RAW_PATH, INTERMEDIATE_PATH, OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)