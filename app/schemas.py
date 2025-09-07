from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class MetaItem(BaseModel):
    id: str
    modality: str                 # "text" | "image" | "video"
    source: Optional[str] = None  # "post" | "comment"
    post_id: Optional[str] = None

    # Text/image payload fields
    text_snippet: Optional[str] = None
    image_url: Optional[str] = None
    cache_path: Optional[str] = None

    # Provenance/time
    timestamp: Optional[str] = None
    platform: Optional[str] = None
    reactions: Optional[Dict[str, Any]] = None

    # Video/frame (if you add later)
    frame_time: Optional[float] = None

    # NEW: chunk metadata for text
    parent_doc_id: Optional[str] = None   # e.g., "post:1F25nYCf0A8" or "comment:91495"
    chunk_id: Optional[str] = None        # e.g., "chunk-0003"
    chunk_start: Optional[int] = None     # token start (approx)
    chunk_end: Optional[int] = None       # token end (approx)


class ChartSpec(BaseModel):
    dataset: str                  # "comments" | "posts"
    chart_type: str               # "bar" | "line" | "scatter"
    x: str                        # e.g., "date"
    y: Optional[str] = None       # numeric column or "count"
    group_by: Optional[str] = None
    time_bin: Optional[str] = None  # day|week|month
    filters: Optional[List[Dict[str, Any]]] = None

class AgentResponse(BaseModel):
    answer: str
    evidence_ids: List[str]
    image_ids: Optional[List[str]] = None
    image_captions: Optional[List[str]] = None
    chart_spec: Optional[ChartSpec] = None
