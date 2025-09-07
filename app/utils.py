import re, json
from typing import Any, Dict, List, Optional

HASHTAG_RE = re.compile(r"(#[\w\d_]+)", re.UNICODE)

def safe_json_parse(s: Optional[str]) -> Any:
    if not s: return None
    try:
        return json.loads(s)
    except Exception:
        try:
            # some rows looked like python-ish repr for lists
            return eval(s, {}, {})
        except Exception:
            return None

def extract_hashtags(text: Optional[str]) -> List[str]:
    if not text: return []
    return [h.lower() for h in HASHTAG_RE.findall(text)]

def first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return None
