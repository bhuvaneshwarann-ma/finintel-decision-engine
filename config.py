import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILINGS_DIR = DATA_DIR / "corporate_filings"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AI_MODE = os.getenv("AI_MODE", "mock").lower()
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Whitelisted items for security & boundary validation
VALID_TICKERS = ["TATAMOTORS", "INFOSYS", "XYZ_CORP"]
VALID_PERSONAS = ["conservative", "aggressive"]
VALID_SCENARIOS = ["aligned", "conflict", "degraded", "stale_behavioral"]

# Staleness threshold for filings (months)
FILING_STALENESS_MONTHS = 12
