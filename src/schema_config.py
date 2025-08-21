# src/schema_config.py
# Referência: Dicionário de Dados ClickBus (valores artificiais: '0' e '1').

# --- Colunas origem (raw) ---
DATE_COLS = ["date_purchase"]
TIME_COLS = ["time_purchase"]

GMV_COL = "gmv_success"
TICKETS_COL = "total_tickets_quantity_success"
CONTACT_ID = "fk_contact"

PRIMARY_KEY = "nk_ota_localizer_id"
PRIMARY_KEY_ALIASES = ["nk_ota_localizer_id", "nk_order_id"]  # cobre variação do slide

ORIGIN_OUT = "place_origin_departure"
DEST_OUT   = "place_destination_departure"
ORIGIN_RET = "place_origin_return"
DEST_RET   = "place_destination_return"
COMPANY_OUT_COL = "fk_departure_ota_bus_company"
COMPANY_RET_COL = "fk_return_ota_bus_company"

# --- Derivados (pipeline) ---
DATETIME_COL = "purchase_datetime"
DATE_FALLBACK_COL = "date_purchase"
WEEKEND_COL = "is_weekend"
HOUR_COL = "hour_of_day"
HAS_RETURN_COL = "has_return"
ROUTE_OUT_COL = "route_out"
ROUTE_RET_COL = "route_return"

# --- Nulos artificiais (dicionário + tolerância robusta) ---
ARTIFICIAL_NULLS = {
    "place_origin_return": ["0", "", "NA", "None", "null"],
    "place_destination_return": ["0", "", "NA", "None", "null"],
    "fk_return_ota_bus_company": ["1", "", "NA", "None", "null"],
}

# --- Tipos sugeridos para read_csv ---
READ_DTYPES = {
    PRIMARY_KEY: "string",
    CONTACT_ID: "string",
    ORIGIN_OUT: "string",
    DEST_OUT: "string",
    ORIGIN_RET: "string",
    DEST_RET: "string",
    COMPANY_OUT_COL: "string",
    COMPANY_RET_COL: "string",
    GMV_COL: "float64",
    TICKETS_COL: "Int64",
}
PARSE_DATES = ["date_purchase"]

# --- Validações úteis em outras etapas ---
REQUIRED_MIN = [*DATE_COLS, *TIME_COLS, GMV_COL, TICKETS_COL]
OPTIONAL_RAW = [ORIGIN_OUT, DEST_OUT, ORIGIN_RET, DEST_RET, COMPANY_OUT_COL, COMPANY_RET_COL]
DERIVED_COLS = [DATETIME_COL, WEEKEND_COL, HOUR_COL, HAS_RETURN_COL, ROUTE_OUT_COL, ROUTE_RET_COL]
