# Mapeamento dinâmico baseado no dicionário de dados fornecido (ClickBus)
# Fonte: Dicionário de Dados PDF

# Colunas de datas e horas
DATE_COLS = ["date_purchase"]
TIME_COLS = ["time_purchase"]

# Colunas numéricas
NUMERIC_COLS = {
    "gmv_success": "float",
    "total_tickets_quantity_success": "int",
}

# Colunas que usam valores artificiais como "0" ou "1" para significar NULL
ARTIFICIAL_NULLS = {
    "place_origin_return": ["0"],
    "place_destination_return": ["0"],
    "fk_return_ota_bus_company": ["1"],
}

# Coluna de ID de compra (para duplicados)
PRIMARY_KEY = "nk_ota_localizer_id"

# Regras de negócio
ORIGIN_OUT = "place_origin_departure"
DEST_OUT = "place_destination_departure"
ORIGIN_RET = "place_origin_return"
DEST_RET = "place_destination_return"

# Todas as referências de colunas ficam aqui.
# (fácil de alterar se o dicionário mudar)

# Datas/horários
DATE_COLS = ["date_purchase"]
TIME_COLS = ["time_purchase"]

# Numéricos (nome -> tipo)
NUMERIC_COLS = {
    "gmv_success": "float",
    "total_tickets_quantity_success": "int",
}

# Convencional: “0”/“1” representando ausência
ARTIFICIAL_NULLS = {
    "place_origin_return": ["0"],
    "place_destination_return": ["0"],
    "fk_return_ota_bus_company": ["1"],
}

# Chaves e campos centrais
PRIMARY_KEY = "nk_ota_localizer_id"
GMV_COL = "gmv_success"
TICKETS_COL = "total_tickets_quantity_success"

# Rotas (ida/volta)
ORIGIN_OUT = "place_origin_departure"
DEST_OUT   = "place_destination_departure"
ORIGIN_RET = "place_origin_return"
DEST_RET   = "place_destination_return"

# Colunas centrais usadas nas features
CONTACT_ID = "fk_contact"
DATETIME_COL = "purchase_datetime"      # gerada no cleaning
DATE_FALLBACK_COL = "date_purchase"     # fallback se necessário

HAS_RETURN_COL = "has_return"
ROUTE_OUT_COL = "route_out"
COMPANY_OUT_COL = "fk_departure_ota_bus_company"  # viação na ida

WEEKEND_COL = "is_weekend"
HOUR_COL = "hour_of_day"