# UHIP config – edit for your city
CITY = "Delhi"
BOUNDS = {  # approx bbox for Delhi NCR
    "min_lon": 76.84, "min_lat": 28.4,
    "max_lon": 77.35, "max_lat": 28.88
}
TARGET_RESOLUTION = 30  # meters
CRS = "EPSG:4326"
DATE_RANGE = ("2024-04-01", "2024-05-31")  # pre-monsoon heat
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
