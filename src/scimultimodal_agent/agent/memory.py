from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Recover the first valid JSON object from model output."""
    text = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    start = fenced.find("{")
    if start >= 0:
        try:
            value, _ = decoder.raw_decode(fenced[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"raw_output": text}


def compact_memory(memory: dict[str, Any]) -> str:
    """Serialize only useful chart evidence for the answer stage."""
    return json.dumps(memory, ensure_ascii=False, separators=(",", ":"))
