# app/search.py - Fixed for Windows paths
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

def _safe_read_index(path):
    """Safely read FAISS index with better error handling"""
    if not path:
        print(f"Path is None or empty")
        return None
    
    # Convert Path object to string if needed
    path_str = str(path)
    
    if not os.path.exists(path_str):
        print(f"FAISS index file not found: {path_str}")
        return None
    
    try:
        print(f"Attempting to load FAISS index from: {path_str}")
        index = faiss.read_index(path_str)
        print(f"Successfully loaded FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        print(f"Error loading FAISS index from {path_str}: {e}")
        return None

class Retriever:
    def __init__(self):
        print("Initializing Retriever...")
        
        # Load metadata
        self.metas: List[MetaItem] = []
        meta_path = str(META_JSONL)  # Ensure it's a string
        print(f"Loading metadata from: {meta_path}")
        
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.metas.append(MetaItem.model_validate_json(line))
            print(f"Loaded {len(self.metas)} metadata items")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metas = []

        # Load FAISS indexes
        print(f"Loading text index from: {FAISS_TEXT}")
        self.text_index = _safe_read_index(FAISS_TEXT)
        
        print(f"Loading image index from: {FAISS_IMAGE}")
        self.image_index = _safe_read_index(FAISS_IMAGE)

        # Create index mappings
        self.text_meta_idxs = [i for i, m in enumerate(self.metas) if m.modality == "text"]
        self.image_meta_idxs = [i for i, m in enumerate(self.metas) if m.modality == "image"]
        
        print(f"Text items: {len(self.text_meta_idxs)}")
        print(f"Image items: {len(self.image_meta_idxs)}")

        # Load CLIP model
        try:
            print(f"Loading CLIP model: {CLIP_MODEL} on {CLIP_DEVICE}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=CLIP_DEVICE
            )
            self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
            self.model.eval()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.model = None
            self.tokenizer = None
            self.preprocess = None

    def embed_text(self, text: str) -> np.ndarray:
        if not self.model or not self.tokenizer:
            raise RuntimeError("CLIP model not loaded")
        
        toks = self.tokenizer([text])
        with torch.no_grad():
            feats = self.model.encode_text(toks.to(CLIP_DEVICE))
        return l2norm(feats.cpu().float().numpy())

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        if not self.model or not self.preprocess:
            return None
        
        try:
            im = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        
        with torch.no_grad():
            feats = self.model.encode_image(self.preprocess(im).unsqueeze(0).to(CLIP_DEVICE))
        return l2norm(feats.cpu().float().numpy())

    def search_text_topk(self, query: str, k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        if self.text_index is None:
            print("Text index is None - cannot perform text search")
            return []
        
        if not self.text_meta_idxs:
            print("No text items available")
            return []
        
        try:
            q = self.embed_text(query)
            D, I = self.text_index.search(q, min(k, len(self.text_meta_idxs)))
            D, I = D[0], I[0]
            
            out = []
            for d, i in zip(D, I):
                if i < len(self.text_meta_idxs):
                    meta_idx = self.text_meta_idxs[i]
                    if meta_idx < len(self.metas):
                        meta = self.metas[meta_idx]
                        out.append({"score": float(d), "meta": meta})
            
            return out
        except Exception as e:
            print(f"Error in text search: {e}")
            return []

    def search_image_topk(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        k: int = TOP_K_DEFAULT
    ) -> List[Dict[str, Any]]:
        if self.image_index is None:
            print("Image index is None - cannot perform image search")
            return []
        
        if not self.image_meta_idxs:
            print("No image items available")
            return []
        
        try:
            if query_text:
                q = self.embed_text(query_text)
            elif query_image_path:
                q = self.embed_image(query_image_path)
                if q is None: 
                    return []
            else:
                return []

            D, I = self.image_index.search(q, min(k, len(self.image_meta_idxs)))
            D, I = D[0], I[0]
            
            out = []
            for d, i in zip(D, I):
                if i < len(self.image_meta_idxs):
                    meta_idx = self.image_meta_idxs[i]
                    if meta_idx < len(self.metas):
                        meta = self.metas[meta_idx]
                        out.append({"score": float(d), "meta": meta})
            
            return out
        except Exception as e:
            print(f"Error in image search: {e}")
            return []

    def search_mixed_topk(
        self,
        query: str,
        k: int = TOP_K_DEFAULT,
        text_take: Optional[int] = None,
        image_take: Optional[int] = None,
        modality_boosts: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        modality_boosts = modality_boosts or MODALITY_BOOSTS
        text_take = text_take or max(3, k // 2)
        image_take = image_take or max(2, k // 3)

        text_hits = self.search_text_topk(query, text_take) if self.text_index else []
        image_hits = self.search_image_topk(query_text=query, k=image_take) if self.image_index else []

        pool: Dict[str, Dict[str, Any]] = {}
        for h in text_hits + image_hits:
            mid = h["meta"].id
            s = h["score"] + modality_boosts.get(h["meta"].modality, 0.0)
            if mid not in pool or s > pool[mid]["score"]:
                pool[mid] = {"score": s, "meta": h["meta"]}

        ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:k]
        return ranked
