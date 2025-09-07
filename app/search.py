import os, json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from PIL import Image
import torch
import open_clip

from .config import (
    FAISS_TEXT, FAISS_IMAGE, META_JSONL,
    CLIP_MODEL, CLIP_PRETRAINED, CLIP_DEVICE,
    TOP_K_DEFAULT, MODALITY_BOOSTS,
)
from .schemas import MetaItem

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return (x / n).astype("float32")

def _safe_read_index(path: str):
    if not os.path.exists(path): return None
    try:
        return faiss.read_index(path)
    except Exception:
        return None

class Retriever:
    def __init__(self):
        self.metas: List[MetaItem] = []
        with open(META_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(MetaItem.model_validate_json(line.strip()))

        self.text_index  = _safe_read_index(FAISS_TEXT)
        self.image_index = _safe_read_index(FAISS_IMAGE)

        self.text_meta_idxs  = [i for i, m in enumerate(self.metas) if m.modality == "text"]
        self.image_meta_idxs = [i for i, m in enumerate(self.metas) if m.modality == "image"]

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=CLIP_DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        toks = self.tokenizer([text])
        with torch.no_grad():
            feats = self.model.encode_text(toks.to(CLIP_DEVICE))
        return l2norm(feats.cpu().float().numpy())

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            im = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        with torch.no_grad():
            feats = self.model.encode_image(self.preprocess(im).unsqueeze(0).to(CLIP_DEVICE))
        return l2norm(feats.cpu().float().numpy())

    def search_text_topk(self, query: str, k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        if self.text_index is None:
            return []
        q = self.embed_text(query)
        D, I = self.text_index.search(q, k)
        D, I = D[0], I[0]
        out = []
        for d, i in zip(D, I):
            meta = self.metas[self.text_meta_idxs[i]]
            out.append({"score": float(d), "meta": meta})
        return out

    def search_image_topk(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        k: int = TOP_K_DEFAULT
    ) -> List[Dict[str, Any]]:
        if self.image_index is None:
            return []
        if query_text:
            q = self.embed_text(query_text)
        elif query_image_path:
            q = self.embed_image(query_image_path)
            if q is None: return []
        else:
            return []

        D, I = self.image_index.search(q, k)
        D, I = D[0], I[0]
        out = []
        for d, i in zip(D, I):
            meta = self.metas[self.image_meta_idxs[i]]
            out.append({"score": float(d), "meta": meta})
        return out

    def search_mixed_topk(
        self,
        query: str,
        k: int = TOP_K_DEFAULT,
        text_take: Optional[int] = None,
        image_take: Optional[int] = None,
        modality_boosts: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        modality_boosts = modality_boosts or MODALITY_BOOSTS
        text_take  = text_take  or max(3, k // 2)
        image_take = image_take or max(2, k // 3)

        text_hits  = self.search_text_topk(query, text_take) if self.text_index else []
        image_hits = self.search_image_topk(query_text=query, k=image_take) if self.image_index else []

        pool: Dict[str, Dict[str, Any]] = {}
        for h in text_hits + image_hits:
            mid = h["meta"].id
            s   = h["score"] + modality_boosts.get(h["meta"].modality, 0.0)
            if mid not in pool or s > pool[mid]["score"]:
                pool[mid] = {"score": s, "meta": h["meta"]}

        ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:k]
        return ranked
