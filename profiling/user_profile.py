from typing import Dict, Any

PERSONA_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "persona_id": "conservative",
        "name": "Conservative Senior",
        "priority": "Capital Preservation",
        "volatility_tolerance": "LOW",
        "max_single_stock_concentration": 0.15,
        "max_sector_concentration": 0.25,
        "debt_tolerance_ratio": 1.2,
        "portfolio": {
            "TATAMOTORS": 0.15,
            "INFOSYS": 0.20,
            "XYZ_CORP": 0.05,
            "FIXED_INCOME_CASH": 0.60
        },
        "description": "Prefers established balance sheets with low leverage and stable dividends. Strongly penalizes debt and concentration risks."
    },
    "aggressive": {
        "persona_id": "aggressive",
        "name": "Aggressive Gen-Z",
        "priority": "Tactical Growth & Breakouts",
        "volatility_tolerance": "HIGH",
        "max_single_stock_concentration": 0.35,
        "max_sector_concentration": 0.50,
        "debt_tolerance_ratio": 3.0,
        "portfolio": {
            "TATAMOTORS": 0.05,
            "INFOSYS": 0.10,
            "XYZ_CORP": 0.02,
            "HIGH_GROWTH_CASH": 0.83
        },
        "description": "Seeks momentum, growth catalysts, and breakout opportunities with higher volatility and drawdown tolerance."
    }
}

def get_persona_profile(persona_name: str) -> Dict[str, Any]:
    key = persona_name.lower().strip()
    return PERSONA_PROFILES.get(key, PERSONA_PROFILES["conservative"])
