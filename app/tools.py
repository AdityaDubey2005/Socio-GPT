from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import tool
from .search import Retriever
from .charts import make_chart
from .config import TOP_K_DEFAULT
import os
import io
import base64
import json
import requests

# OPTIONAL web search provider: Tavily or SerpAPI.
# Set ONE of these env vars:
#   TAVILY_API_KEY=...
#   SERPAPI_API_KEY=...
TAVILY_ENDPOINT = "https://api.tavily.com/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

class TextSearchInput(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K_DEFAULT)

class ImageSearchInput(BaseModel):
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    top_k: int = Field(default=TOP_K_DEFAULT)

class MixedSearchInput(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K_DEFAULT)

class ChartInput(BaseModel):
    # Make chart_spec optional so we don't crash if the LLM omits it
    chart_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Visualization spec dict, e.g. "
            '{"dataset":"posts","chart_type":"bar","x":"top_hashtag","y":"count","filters":[],"time_bin":null}'
        ),
    )
    # Allow natural language fallback if chart_spec is missing
    nl_request: Optional[str] = Field(
        default=None,
        description="Natural-language chart request (e.g., 'plot top hashtags over time'). Used to infer a default spec."
    )

def _infer_spec_from_nl(nl: Optional[str]) -> Dict[str, Any]:
    """Very simple heuristic to avoid crashes if the model omits chart_spec."""
    s = (nl or "").lower()
    # Common intents
    if "hashtag" in s:
        return {
            "dataset": "posts",
            "chart_type": "bar",
            "x": "top_hashtag",
            "y": "count",
            "filters": [],
            "time_bin": None,
            "title": "Top hashtags (posts)"
        }
    if any(k in s for k in ["over time", "trend", "daily", "weekly", "monthly", "timeline", "temporal"]):
        return {
            "dataset": "comments",
            "chart_type": "line",
            "x": "date",
            "y": "count",
            "filters": [],
            "time_bin": "D",
            "title": "Comment volume over time"
        }
    # Safe default
    return {
        "dataset": "posts",
        "chart_type": "bar",
        "x": "top_hashtag",
        "y": "count",
        "filters": [],
        "time_bin": None,
        "title": "Top hashtags (posts)"
    }


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5

def _pack_results(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Compact structure for the agent, not raw MetaItem objects
    out = []
    for h in hits:
        meta = h["meta"]
        out.append({
            "id": meta.id,
            "modality": meta.modality,
            "source": meta.source,
            "post_id": meta.post_id,
            "text_snippet": meta.text_snippet,
            "image_url": meta.image_url,
            "cache_path": meta.cache_path,
            "timestamp": meta.timestamp,
            "parent_doc_id": getattr(meta, "parent_doc_id", None),
            "chunk_id": getattr(meta, "chunk_id", None),
        })
    return {"results": out}

def _b64_image(path_or_url: str) -> Optional[str]:
    try:
        if path_or_url.startswith("http"):
            b = requests.get(path_or_url, timeout=15).content
        else:
            with open(path_or_url, "rb") as f:
                b = f.read()
        return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return None

class ToolBinder:
    """Holds a retriever instance and exposes LangChain tools bound to it."""
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @tool("text_search", args_schema=TextSearchInput, return_direct=False)
    def text_search(query: str, top_k: int = TOP_K_DEFAULT) -> str:
        """Search the dataset text chunks first. Returns JSON with 'results'."""
        hits = ToolBinder._self().retriever.search_text_topk(query, top_k)
        return json.dumps(_pack_results(hits))

    @tool("image_search", args_schema=ImageSearchInput, return_direct=False)
    def image_search(query_text: Optional[str] = None, query_image_path: Optional[str] = None, top_k: int = TOP_K_DEFAULT) -> str:
        """Search the dataset images by text or by uploaded image. Returns JSON with 'results'."""
        hits = ToolBinder._self().retriever.search_image_topk(query_text, query_image_path, top_k)
        return json.dumps(_pack_results(hits))

    @tool("mixed_search", args_schema=MixedSearchInput, return_direct=False)
    def mixed_search(query: str, top_k: int = TOP_K_DEFAULT) -> str:
        """Search both text and images and blend results. Returns JSON with 'results'."""
        hits = ToolBinder._self().retriever.search_mixed_topk(query, top_k)
        return json.dumps(_pack_results(hits))

    @tool("chart_tool", args_schema=ChartInput, return_direct=False)
    def chart_tool(chart_spec: Optional[Dict[str, Any]] = None, nl_request: Optional[str] = None) -> str:
        """
        Render a chart from dataset features.
        Prefer passing a complete 'chart_spec'. If omitted, a simple default is inferred from 'nl_request'.
        Returns JSON with:
        - ok: bool
        - echo_spec: the spec used
        - png_b64: the chart image as base64 (if ok)
        - error: message (if not ok)
        """
        try:
            spec = chart_spec or _infer_spec_from_nl(nl_request)
            fig = make_chart(spec)  # validates/plots using your parquet features
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return json.dumps({"ok": True, "png_b64": b64, "echo_spec": spec})
        except Exception as e:
            # Never raise—return a payload the LLM can read and correct on next try
            return json.dumps({"ok": False, "error": str(e)})
        
    @tool("web_search", args_schema=WebSearchInput, return_direct=False)
    def web_search(query: str, max_results: int = 5) -> str:
        """Fallback web search (Tavily or SerpAPI). Returns JSON with simplified results."""
        tav_key = os.getenv("TAVILY_API_KEY")
        serp_key = os.getenv("SERPAPI_API_KEY")
        items = []
        try:
            if tav_key:
                resp = requests.post(TAVILY_ENDPOINT, json={"api_key": tav_key, "query": query, "max_results": max_results}, timeout=20)
                data = resp.json()
                for r in data.get("results", [])[:max_results]:
                    items.append({"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")})
            elif serp_key:
                params = {"engine": "google", "q": query, "api_key": serp_key}
                data = requests.get(SERPAPI_ENDPOINT, params=params, timeout=20).json()
                for r in (data.get("organic_results") or [])[:max_results]:
                    items.append({"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")})
            else:
                items.append({"title":"No web key configured","url":"","snippet":"Set TAVILY_API_KEY or SERPAPI_API_KEY"})
        except Exception as e:
            items.append({"title":"Web search error","url":"","snippet":str(e)})
        return json.dumps({"results": items})

    # trick to access bound instance inside static tool functions
    _BOUND: "ToolBinder" = None
    @classmethod
    def bind(cls, retriever: Retriever):
        cls._BOUND = cls(retriever)
        return cls._BOUND
    @classmethod
    def _self(cls) -> "ToolBinder":
        return cls._BOUND
