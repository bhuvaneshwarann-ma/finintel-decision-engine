from fastapi import HTTPException
from config import VALID_TICKERS, VALID_PERSONAS, VALID_SCENARIOS

# Banned clinical/diagnostic terms per §18.5 and Test 22
BANNED_DIAGNOSTIC_TERMS = [
    "disorder", "addiction", "addicted", "pathological", "illness",
    "compulsive", "bipolar", "depressed", "depression", "schizophrenic",
    "insane", "psychosis", "neurotic", "syndrome", "affliction", "disease"
]

def validate_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if cleaned not in VALID_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker '{ticker}'. Supported tickers: {VALID_TICKERS}"
        )
    return cleaned

def validate_persona(persona: str) -> str:
    cleaned = persona.strip().lower()
    if cleaned not in VALID_PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid persona '{persona}'. Supported personas: {VALID_PERSONAS}"
        )
    return cleaned

def validate_scenario(scenario: str) -> str:
    if not scenario:
        return "aligned"
    cleaned = scenario.strip().lower()
    if cleaned not in VALID_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario '{scenario}'. Supported scenarios: {VALID_SCENARIOS}"
        )
    return cleaned

def check_no_diagnostic_terms(text: str) -> bool:
    """
    Returns True if text contains no banned diagnostic/clinical words.
    """
    lower_text = text.lower()
    for term in BANNED_DIAGNOSTIC_TERMS:
        if term in lower_text:
            return False
    return True
