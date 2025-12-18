from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# File paths
RESTAURANTS_FILE = RAW_DATA_DIR / "restaurants.csv"
ITEMS_FILE = RAW_DATA_DIR / "items.csv"
SALES_FILE = RAW_DATA_DIR / "sales.csv"
