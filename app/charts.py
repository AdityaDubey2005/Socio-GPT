# app/charts.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt

from .config import FEATURE_POSTS, FEATURE_COMMENTS

def _load_dataset(name: str) -> pd.DataFrame:
    if name == "posts":
        return pd.read_parquet(FEATURE_POSTS)
    if name == "comments":
        return pd.read_parquet(FEATURE_COMMENTS)
    raise ValueError(f"Unknown dataset: {name}")

def _apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        col = f.get("col")
        op  = f.get("op", "eq")
        val = f.get("val")
        if col not in out.columns:
            continue
        if op == "eq":
            out = out[out[col] == val]
        elif op == "neq":
            out = out[out[col] != val]
        elif op == "contains":
            out = out[out[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op == "in":
            out = out[out[col].isin(val if isinstance(val, list) else [val])]
        elif op == "notin":
            out = out[~out[col].isin(val if isinstance(val, list) else [val])]
        # add more ops if needed
    return out

def _maybe_timebin(df: pd.DataFrame, time_col: str, time_bin: Optional[str]) -> pd.DataFrame:
    if not time_bin:
        return df
    if time_col not in df.columns:
        return df
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col])
    # Simple resample by time_bin string (e.g., 'D', 'W', 'M')
    grp = tmp.set_index(time_col).resample(time_bin).size().reset_index(name="count")
    grp.rename(columns={time_col: "time"}, inplace=True)
    return grp

def make_chart(spec: Dict[str, Any]):
    """
    Minimal chart factory used by the app:
    spec = {
      "dataset": "posts" | "comments",
      "chart_type": "bar" | "line",
      "x": "<col>",               # for bar: category, for line: time or category
      "y": "<col or 'count'>",
      "filters": [ {col, op, val}, ... ],
      "time_bin": null | "D" | "W" | "M",
      "title": "optional title"
    }
    """
    ds = spec.get("dataset", "posts")
    ctype = spec.get("chart_type", "bar")
    xcol = spec.get("x")
    ycol = spec.get("y", "count")
    filters = spec.get("filters", [])
    time_bin = spec.get("time_bin")
    title = spec.get("title") or ""

    df = _load_dataset(ds)
    df = _apply_filters(df, filters)

    if ctype == "line" and time_bin:
        # time series count over time
        time_col = "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if not time_col:
            raise ValueError("No time column found for time series")
        ts = _maybe_timebin(df, time_col=time_col, time_bin=time_bin)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts["time"], ts["count"])
        ax.set_xlabel("time")
        ax.set_ylabel("count")
        ax.set_title(title or f"{ds} over time")
        fig.tight_layout()
        return fig

    # Aggregations for bar/line over categories
    if ycol == "count":
        if xcol not in df.columns:
            raise ValueError(f"x column '{xcol}' not in dataset")
        agg = df.groupby(xcol).size().reset_index(name="count").sort_values("count", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 4))
        if ctype == "bar":
            ax.bar(agg[xcol].astype(str), agg["count"])
        else:
            ax.plot(agg[xcol].astype(str), agg["count"])
        ax.set_xlabel(xcol)
        ax.set_ylabel("count")
        ax.set_title(title or f"Top {xcol} by count")
        ax.tick_params(axis='x', rotation=35)
        fig.tight_layout()
        return fig

    # If y is a numeric column
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"Columns missing: x='{xcol}' y='{ycol}'")
    agg = df.groupby(xcol)[ycol].mean().reset_index().sort_values(ycol, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 4))
    if ctype == "bar":
        ax.bar(agg[xcol].astype(str), agg[ycol])
    else:
        ax.plot(agg[xcol].astype(str), agg[ycol])
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title or f"{ycol} by {xcol}")
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    return fig
