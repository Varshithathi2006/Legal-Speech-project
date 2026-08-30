import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Setup Paths & Load Old Vector Store
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(".")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "legal_rag_vectorstore")

if not os.path.exists(VECTORSTORE_DIR):
    raise FileNotFoundError(f"Vectorstore not found at {VECTORSTORE_DIR}")

print("======================================================================")
print("             STEP 1: DIAGNOSE CURRENT RETRIEVAL QUALITY               ")
print("======================================================================")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = chroma_client.get_collection(name="legal_speech_rag")

print(f"Total Chunks in Collection: {collection.count()}\n")

sample_queries = [
    "What arguments were raised regarding Article 21 and framing of charges under Representation of People Act?",
    "What are the statutory grounds for setting aside an arbitral award under Section 34?",
    "What did the court discuss regarding framing of charges and disqualification from elections?",
    "What are the requirements for bail under CrPC and fundamental rights under Article 14?"
]

for q_idx, query in enumerate(sample_queries, 1):
    print(f"\n{'='*70}")
    print(f"QUERY {q_idx}: {query}")
    print(f"{'='*70}")
    
    query_vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=4)
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    for c_idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        sim_score = max(0.0, round(1.0 - float(dist), 4)) if dist <= 1.0 else round(1.0 / (1.0 + float(dist)), 4)
        word_count = len(doc.split())
        
        print(f"\n--- Chunk {c_idx} [Similarity Score: {sim_score} | Words: {word_count}] ---")
        print(f"Source Type : {meta.get('source_type')}")
        if meta.get("source_type") == "speech_transcript":
            print(f"Case Name   : {meta.get('case_name')}")
            print(f"Track       : {meta.get('track')}")
            print(f"Video ID    : {meta.get('video_id')}")
            print(f"Timestamp   : [{meta.get('start_time'):.1f}s - {meta.get('end_time'):.1f}s]")
            print(f"Speakers    : {meta.get('speakers')}")
        else:
            print(f"Act Name    : {meta.get('act_name')}")
            print(f"Section ID  : {meta.get('section_id')}")
            print(f"Title       : {meta.get('title')}")
            
        print("\nContent Snippet:")
        print(doc.strip()[:350] + ("..." if len(doc) > 350 else ""))

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY FOR STEP 1 COMPLETE")
print("="*70)
