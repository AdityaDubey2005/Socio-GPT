# app/ingest.py
import os, json, hashlib, math
from typing import List, Dict, Any, Tuple, Optional
import requests
from io import BytesIO

import numpy as np
import pandas as pd
import faiss
from PIL import Image
import torch
import open_clip
from tqdm.auto import tqdm

from .config import (
    POSTS_CSV, COMMENTS_CSV,
    CACHE_DIR, INDEX_DIR, META_JSONL, MANIFEST_JSON,
    FEATURE_COMMENTS, FEATURE_POSTS,
    FAISS_TEXT, FAISS_IMAGE,
    CLIP_MODEL, CLIP_PRETRAINED, CLIP_DEVICE,
)

from .utils import safe_json_parse, extract_hashtags, first_non_empty

# ------------------------------
# CPU-friendly knobs
# ------------------------------
TEXT_BATCH = int(os.getenv("MM_TEXT_BATCH", "64"))      # 64 is safe for 16GB RAM on CPU
IMAGE_BATCH = int(os.getenv("MM_IMAGE_BATCH", "16"))
TIMEOUT = (10, 20)  # (connect, read) seconds for image downloads

# Optional quick-run limits (None = full dataset)
LIMIT_POSTS = os.getenv("MM_LIMIT_POSTS")
LIMIT_POSTS = int(LIMIT_POSTS) if LIMIT_POSTS else None
LIMIT_COMMENTS = os.getenv("MM_LIMIT_COMMENTS")
LIMIT_COMMENTS = int(LIMIT_COMMENTS) if LIMIT_COMMENTS else None

# ------------------------------
# Helpers
# ------------------------------
def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return (x / n).astype("float32")

def ensure_parent(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def fetch_image_to_cache(url: str) -> Optional[str]:
    try:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        local = os.path.join(CACHE_DIR, "images", f"{h}.jpg")
        ensure_parent(local)
        if not os.path.exists(local):
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            Image.open(BytesIO(r.content)).convert("RGB").save(local, "JPEG", quality=92)
        return local
    except Exception:
        return None

def derive_youtube_thumb(link: str) -> str:
    if isinstance(link, str) and "youtube.com/watch?v=" in link:
        vid = link.split("v=")[-1].split("&")[0]
        return f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"
    return ""

def parse_sentiment(obj):
    if not obj: return None
    try:
        s = obj.get("Sentiment") or obj.get("sentiment")
        if isinstance(s, dict) and s:
            return max(s, key=s.get)
        return None
    except Exception:
        return None

# ---------- TEXT CHUNKING (word-based, fast & safe) ----------
CHUNK_WORDS = 60       # ~ chunk size (maps well to CLIP’s ~77-token cap)
OVERLAP_WORDS = 20
STEP = max(1, CHUNK_WORDS - OVERLAP_WORDS)

def split_text_to_chunks(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple sliding window by words.
    Returns list of (chunk_text, start_word_idx, end_word_idx [exclusive])
    """
    if not text or not str(text).strip():
        return []
    words = str(text).split()
    n = len(words)
    chunks: List[Tuple[str, int, int]] = []
    for start in range(0, n, STEP):
        end = min(start + CHUNK_WORDS, n)
        if start >= end:
            break
        sub = " ".join(words[start:end]).strip()
        if sub:
            chunks.append((sub, start, end))
        if end == n:
            break
    return chunks

# ------------------------------
# MAIN
# ------------------------------
def main():
    # Keep torch from oversubscribing CPU threads
    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load CSVs (optionally limited)
    print("[ingest] Loading CSVs…")
    posts_df    = pd.read_csv(POSTS_CSV)
    comments_df = pd.read_csv(COMMENTS_CSV)

    if LIMIT_POSTS is not None:
        posts_df = posts_df.head(LIMIT_POSTS).copy()
    if LIMIT_COMMENTS is not None:
        comments_df = comments_df.head(LIMIT_COMMENTS).copy()

    print(f"[ingest] posts: {len(posts_df):,} | comments: {len(comments_df):,}")

    # Normalize dates
    if "timestamp" in posts_df.columns:
        posts_df["date"] = pd.to_datetime(posts_df["timestamp"], errors="coerce")
    if "date_of_comment" in comments_df.columns:
        comments_df["date"] = pd.to_datetime(comments_df["date_of_comment"], errors="coerce")

    # Final text field
    posts_df["text_final"] = posts_df.apply(
        lambda r: first_non_empty(r.get("text"), r.get("raw_text")), axis=1
    )
    comments_df["text_final"] = comments_df.apply(
        lambda r: first_non_empty(r.get("text"), r.get("raw_text")), axis=1
    )

    # Light features for charting
    print("[ingest] Deriving light features for charts…")
    posts_df["hashtags"]    = posts_df["text_final"].apply(extract_hashtags)
    comments_df["hashtags"] = comments_df["text_final"].apply(extract_hashtags)
    posts_df["top_hashtag"]    = posts_df["hashtags"].apply(lambda hs: hs[0] if hs else None)
    comments_df["top_hashtag"] = comments_df["hashtags"].apply(lambda hs: hs[0] if hs else None)

    if "text_analysis" in comments_df.columns:
        comments_df["text_analysis_obj"] = comments_df["text_analysis"].apply(safe_json_parse)
        comments_df["sentiment_label"] = comments_df["text_analysis_obj"].apply(parse_sentiment)
    if "text_analysis" in posts_df.columns:
        posts_df["text_analysis_obj"] = posts_df["text_analysis"].apply(safe_json_parse)
        posts_df["sentiment_label"] = posts_df["text_analysis_obj"].apply(parse_sentiment)

    # Save features for charting
    posts_df.to_parquet(FEATURE_POSTS, index=False)
    comments_df.to_parquet(FEATURE_COMMENTS, index=False)
    print(f"[ingest] Saved features:\n  - {FEATURE_POSTS}\n  - {FEATURE_COMMENTS}")

    # CLIP model (CPU)
    print(f"[ingest] Loading CLIP model: {CLIP_MODEL} ({CLIP_PRETRAINED}) on {CLIP_DEVICE}…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=CLIP_DEVICE
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    print("[ingest] CLIP ready ✅")

    # ---------- Embedding helpers ----------
    def embed_text_batch(texts: List[str]) -> np.ndarray:
        toks = tokenizer(texts)
        with torch.no_grad():
            feats = model.encode_text(toks.to(CLIP_DEVICE))
        return l2norm(feats.cpu().float().numpy())

    def embed_image_batch(paths: List[str]) -> np.ndarray:
        ims = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
                ims.append(preprocess(im))
            except Exception:
                # skip broken images
                pass
        if not ims:
            return np.zeros((0, 512), dtype="float32")  # ViT-B-32 => 512-d
        ims_t = torch.stack(ims, dim=0)
        with torch.no_grad():
            feats = model.encode_image(ims_t.to(CLIP_DEVICE))
        return l2norm(feats.cpu().float().numpy())

    # ---------- TEXT: chunk + meta + FAISS (incremental) ----------
    print("[ingest] Chunking post texts…")
    metas: List[Dict[str, Any]] = []
    text_payloads: List[str] = []

    for _, r in tqdm(posts_df.iterrows(), total=len(posts_df), desc="Text chunks (posts)", unit="post"):
        pid = r.get("post_id") or r.get("id")
        t   = r.get("text_final")
        if not t: 
            continue
        for idx, (chunk_text, wstart, wend) in enumerate(split_text_to_chunks(t), start=1):
            cid = f"chunk-{idx:04d}"
            metas.append({
                "id": f"text:post:{pid}:{cid}",
                "modality": "text",
                "source": "post",
                "post_id": str(pid) if pid is not None else None,
                "parent_doc_id": f"post:{pid}",
                "chunk_id": cid,
                "chunk_start": int(wstart),
                "chunk_end": int(wend),
                "text_snippet": chunk_text[:500],
                "timestamp": r.get("timestamp"),
            })
            text_payloads.append(chunk_text[:2000])

    print("[ingest] Chunking comment texts…")
    for _, r in tqdm(comments_df.iterrows(), total=len(comments_df), desc="Text chunks (comments)", unit="cmt"):
        cid = r.get("comment_id") or r.get("id")
        t   = r.get("text_final")
        if not t:
            continue
        post_id = r.get("post_id")
        for idx, (chunk_text, wstart, wend) in enumerate(split_text_to_chunks(t), start=1):
            chid = f"chunk-{idx:04d}"
            metas.append({
                "id": f"text:comment:{cid}:{chid}",
                "modality": "text",
                "source": "comment",
                "post_id": str(post_id) if post_id is not None else None,
                "parent_doc_id": f"comment:{cid}",
                "chunk_id": chid,
                "chunk_start": int(wstart),
                "chunk_end": int(wend),
                "text_snippet": chunk_text[:500],
                "timestamp": r.get("date_of_comment"),
            })
            text_payloads.append(chunk_text[:2000])

    # Build text FAISS incrementally to avoid big RAM spikes
    print(f"[ingest] Embedding TEXT chunks: {len(text_payloads):,} items (batch={TEXT_BATCH})…")
    idx_text = None
    total_batches = math.ceil(len(text_payloads) / max(1, TEXT_BATCH))
    for bi in tqdm(range(total_batches), desc="Embedding TEXT", unit="batch"):
        start = bi * TEXT_BATCH
        end = min((bi + 1) * TEXT_BATCH, len(text_payloads))
        if start >= end:
            break
        vecs = embed_text_batch(text_payloads[start:end])
        if vecs.shape[0] == 0:
            continue
        if idx_text is None:
            idx_text = faiss.IndexFlatIP(vecs.shape[1])
        idx_text.add(vecs)

    if idx_text is not None:
        ensure_parent(FAISS_TEXT)
        faiss.write_index(idx_text, FAISS_TEXT)
        print(f"[ingest] Wrote text index: {FAISS_TEXT}")
    else:
        print("[ingest] No text to embed (index not created).")

    # ---------- IMAGES: collect + meta + FAISS (incremental) ----------
    print("[ingest] Collecting/downloading images…")
    image_paths: List[str] = []
    for _, r in tqdm(posts_df.iterrows(), total=len(posts_df), desc="Image collection (posts)", unit="post"):
        pid = r.get("post_id") or r.get("id")
        # Figure out image URL
        media_url = None
        mu = r.get("media_url")
        if isinstance(mu, str) and mu:
            obj = safe_json_parse(mu)
            if isinstance(obj, list) and obj:
                media_url = obj[0]
            elif mu.startswith("http"):
                media_url = mu
        if not media_url and isinstance(r.get("link"), str):
            media_url = derive_youtube_thumb(r["link"])
        if not media_url:
            continue

        local = fetch_image_to_cache(media_url)
        if not local:
            continue

        metas.append({
            "id": f"image:post:{pid}",
            "modality": "image",
            "source": "post",
            "post_id": str(pid) if pid is not None else None,
            "image_url": media_url,
            "cache_path": local,
            "text_snippet": (r.get("text_final") or "")[:300],
            "timestamp": r.get("timestamp")
        })
        image_paths.append(local)

    print(f"[ingest] Embedding IMAGES: {len(image_paths):,} items (batch={IMAGE_BATCH})…")
    idx_img = None
    total_ibatches = math.ceil(len(image_paths) / max(1, IMAGE_BATCH))
    for bi in tqdm(range(total_ibatches), desc="Embedding IMAGES", unit="batch"):
        start = bi * IMAGE_BATCH
        end = min((bi + 1) * IMAGE_BATCH, len(image_paths))
        if start >= end:
            break
        batch_paths = image_paths[start:end]
        ivecs = embed_image_batch(batch_paths)
        if ivecs.shape[0] == 0:
            continue
        if idx_img is None:
            idx_img = faiss.IndexFlatIP(ivecs.shape[1])
        idx_img.add(ivecs)

    if idx_img is not None:
        ensure_parent(FAISS_IMAGE)
        faiss.write_index(idx_img, FAISS_IMAGE)
        print(f"[ingest] Wrote image index: {FAISS_IMAGE}")
    else:
        print("[ingest] No images to embed (index not created).")

    # ---------- Write meta.jsonl ----------
    print(f"[ingest] Writing metadata file: {META_JSONL} ({len(metas):,} items)…")
    ensure_parent(META_JSONL)
    with open(META_JSONL, "w", encoding="utf-8") as f:
        for m in tqdm(metas, desc="Write meta.jsonl", unit="item"):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------- Manifest ----------
    manifest = {
        "counts": {
            "total": len(metas),
            "text": sum(1 for m in metas if m["modality"] == "text"),
            "image": sum(1 for m in metas if m["modality"] == "image"),
        },
        "paths": {
            "FAISS_TEXT": FAISS_TEXT,
            "FAISS_IMAGE": FAISS_IMAGE,
            "META_JSONL": META_JSONL,
            "FEATURE_COMMENTS": FEATURE_COMMENTS,
            "FEATURE_POSTS": FEATURE_POSTS
        },
        "chunking": {
            "chunk_words": CHUNK_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "step": STEP
        },
        "batching": {
            "TEXT_BATCH": TEXT_BATCH,
            "IMAGE_BATCH": IMAGE_BATCH
        }
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Ingestion complete ✅")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
