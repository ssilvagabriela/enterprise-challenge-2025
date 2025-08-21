import pandas as pd
import logging
from src.config import RAW_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_purchases(filename: str = "df_t.csv") -> pd.DataFrame:
    """
    Lê o arquivo CSV de compras e retorna um DataFrame.
    
    Args:
        filename (str): nome do arquivo CSV na pasta RAW_PATH.
    Returns:
        pd.DataFrame
    """
    file_path = RAW_PATH / filename
    logging.info(f"Lendo dados de {file_path}...")
    df = pd.read_csv(file_path)
    logging.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas.")
    return df