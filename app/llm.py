import base64, json, os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .config import OPENAI_MODEL, MAX_IMAGES_TO_SEND

client = OpenAI()

SYSTEM = """You are a concise multimodal research assistant.
You get a user query and a set of retrieved evidence (texts and image references).
Rules:
- Be concise (4–8 sentences) and non-repetitive.
- Ground claims in the evidence; include a short bulleted list of evidence IDs at the end.
- Only include a chart_spec JSON when the user explicitly asks to visualize (plot/chart/trend/over time).
- Never invent dataset columns; use realistic ones like date/timestamp/hashtags/sentiment/views/engagement.
Output JSON only:
{
  "answer": "...",
  "evidence_ids": ["..."],
  "image_captions": ["..."],    // optional
  "chart_spec": { ... }         // optional
}
"""

def _as_image_content(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = item.get("image_url")
    path = item.get("cache_path")
    if url:
        return {"type": "image_url", "image_url": {"url": url}}
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    return None

def generate_multimodal_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    query_image_path: Optional[str] = None,
    wants_chart: bool = False,
    max_images: int = MAX_IMAGES_TO_SEND
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": f"User query: {query}"}]

    if query_image_path and os.path.exists(query_image_path):
        with open(query_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    ids: List[str] = []
    sent_images = 0
    for it in evidence:
        if it.get("id"): ids.append(it["id"])
        if it.get("modality") == "image" and sent_images < max_images:
            ic = _as_image_content(it)
            if ic:
                content.append(ic)
                sent_images += 1

    # compact evidence text lines
    lines = []
    for it in evidence[:12]:
        snippet = (it.get("text_snippet") or "")[:160]
        lines.append(f"- {it.get('id')} :: {snippet}")
    if lines:
        content.append({"type": "text", "text": "Evidence:\n" + "\n".join(lines)})

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
        {"role": "user", "content": "Return a compact JSON only."}
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    try:
        obj = json.loads(resp.choices[0].message.content)
    except Exception:
        obj = {"answer": resp.choices[0].message.content}

    obj.setdefault("evidence_ids", list(dict.fromkeys(ids))[:20])
    if not wants_chart and obj.get("chart_spec"):
        obj.pop("chart_spec", None)
    return obj
