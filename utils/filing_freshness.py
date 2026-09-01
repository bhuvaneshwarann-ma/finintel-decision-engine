from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any
from config import FILING_STALENESS_MONTHS

REFERENCE_DATE = date(2026, 2, 15)

def check_filing_freshness(doc_date_str: str, threshold_months: int = FILING_STALENESS_MONTHS) -> Tuple[bool, int, Optional[str]]:
    """
    Evaluates whether a filing date exceeds the staleness threshold.
    Returns: (is_stale, age_months, warning_message)
    """
    if not doc_date_str:
        return True, 999, "Filing date missing or unparseable; treated as unverified freshness."

    try:
        doc_date = datetime.strptime(doc_date_str.strip(), "%Y-%m-%d").date()
    except Exception:
        return True, 999, f"Invalid date format '{doc_date_str}'; expected YYYY-MM-DD."

    diff_days = (REFERENCE_DATE - doc_date).days
    age_months = max(0, int(diff_days / 30.4375))
    
    if age_months > threshold_months:
        warning = (
            f"Filing Freshness Warning: The latest available filing is dated {doc_date_str} "
            f"({age_months} months old, exceeding the {threshold_months}-month freshness threshold). "
            f"Regulatory disclosures may not reflect recent financial condition or off-balance sheet liabilities."
        )
        return True, age_months, warning
    
    return False, age_months, None
