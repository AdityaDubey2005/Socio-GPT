
from app.search import Retriever
r = Retriever()

print("Metas:", len(r.metas))
print("Text index:", "OK" if r.text_index else "MISSING")
print("Image index:", "OK" if r.image_index else "MISSING")

q = "How to make notes with ChatGPT"
print("\nTop-5 TEXT hits:")
for h in r.search_text_topk(q, 5):
    m = h["meta"]
    print(f"  {h['score']:.3f} | {m.id} | {m.modality} | {m.source} | {(m.text_snippet or '')[:80]}")

print("\nTop-5 IMAGE hits (text->image cross-modal):")
for h in r.search_image_topk(query_text=q, k=5):
    m = h["meta"]
    print(f"  {h['score']:.3f} | {m.id} | {m.modality} | {m.image_url or m.cache_path}")
