import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILINGS_DIR = DATA_DIR / "corporate_filings"

# Vercel serverless environment support
IS_VERCEL = bool(os.getenv("VERCEL"))

if IS_VERCEL:
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
    AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "/tmp/auth.db")
else:
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
    AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", str(BASE_DIR / "auth.db"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AI_MODE = os.getenv("AI_MODE", "mock").lower()
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Authentication Configuration (§5)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Whitelisted items for security & boundary validation
VALID_TICKERS = [
    "TATAMOTORS", "INFOSYS", "XYZ_CORP",
    "RELIANCE", "HDFCBANK", "TCS", "BHARTIARTL", "ITC"
]
VALID_PERSONAS = ["conservative", "aggressive"]
VALID_SCENARIOS = ["aligned", "conflict", "degraded", "stale_behavioral"]

# Staleness threshold for filings (months)
FILING_STALENESS_MONTHS = 12
