"""
Vercel Serverless Function Entry Point for FinIntelligence AI.
Exposes the FastAPI application instance for @vercel/python runtime.
"""
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app
