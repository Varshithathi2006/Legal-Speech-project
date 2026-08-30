import os
import sys
import json
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Setup Paths & CUDA Configurations
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(".")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "legal_rag_vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

TRACKS = {
    "moot_court": {
        "transcripts": os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "transcripts"),
        "documents":   os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "documents"),
        "metadata":    os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "metadata", "video_metadata.csv"),
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Step 2 & 3: Re-chunking & Fast BGE Indexing on {device}...", flush=True)

# ──────────────────────────────────────────────────────────────────────
# Load BGE-Base Model (Fast & High Accuracy)
# ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"Loading SentenceTransformer embedding model '{EMBEDDING_MODEL_NAME}' on {device}...", flush=True)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
print(f"BGE-Large embedding model loaded successfully on {device}!", flush=True)

chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
# Delete old collection and recreate fresh to avoid dimension/stale data issues
try:
    chroma_client.delete_collection(name="legal_speech_rag_bge")
    print("Deleted old 'legal_speech_rag_bge' collection for clean rebuild.", flush=True)
except Exception:
    pass
collection = chroma_client.create_collection(
    name="legal_speech_rag_bge",
    metadata={"hnsw:space": "cosine"}
)

print(f"ChromaDB Collection 'legal_speech_rag_bge' initialized at: {VECTORSTORE_DIR}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# Re-chunking Function: Target 150-300 Words Per Chunk
# ──────────────────────────────────────────────────────────────────────
def chunk_transcript_dialogues_bge(video_id, case_name, track_name, segments, target_word_min=150, target_word_max=300):
    chunks = []
    if not segments:
        return chunks

    i = 0
    chunk_counter = 0
    n = len(segments)
    
    while i < n:
        window = []
        curr_words = 0
        j = i
        
        while j < n and curr_words < target_word_max:
            seg = segments[j]
            txt = seg.get("text", "").strip()
            w_len = len(txt.split())
            window.append(seg)
            curr_words += w_len
            j += 1
            if curr_words >= target_word_min:
                break
                
        if not window:
            break
            
        start_time = window[0].get("start", 0.0)
        end_time = window[-1].get("end", 0.0)
        speakers = sorted(list({s.get("speaker", "SPEAKER_00") for s in window}))
        
        lines = []
        for s in window:
            txt = s.get("text", "").strip()
            if txt:
                spk = s.get("speaker", "SPEAKER")
                lines.append(f"[{s.get('start', 0.0):.1f}s-{s.get('end', 0.0):.1f}s] {spk}: {txt}")
        
        combined_text = "\n".join(lines)
        w_total = len(combined_text.split())
        
        if w_total >= 15:
            chunk_id = f"bge_{track_name}_{video_id}_chunk_{chunk_counter}"
            metadata = {
                "chunk_id": chunk_id,
                "source_type": "speech_transcript",
                "track": track_name,
                "video_id": str(video_id),
                "case_name": str(case_name)[:100],
                "start_time": float(start_time),
                "end_time": float(end_time),
                "speakers": ", ".join(speakers),
                "num_turns": len(window),
                "word_count": int(w_total)
            }
            
            chunks.append({
                "id": chunk_id,
                "text": f"Case: {case_name}\nTrack: {track_name}\n" + combined_text,
                "metadata": metadata
            })
            chunk_counter += 1

        overlap_step = max(1, len(window) // 2)
        i += overlap_step

    return chunks

# ──────────────────────────────────────────────────────────────────────
# Ingest & Re-chunk Data
# ──────────────────────────────────────────────────────────────────────
all_documents = []
all_metadatas = []
all_ids = []

processed_docs_count = 0
processed_speech_chunks = 0

for track_name, paths in TRACKS.items():
    print(f"Processing Track: {track_name}...", flush=True)
    
    # 1. Statutory Documents
    docs_dir = paths["documents"]
    if os.path.exists(docs_dir):
        doc_files = [f for f in os.listdir(docs_dir) if f.endswith(".json")]
        for df_name in doc_files:
            df_path = os.path.join(docs_dir, df_name)
            try:
                with open(df_path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
                
                act_name = doc_data.get("act_name", "Statute")
                category = doc_data.get("category", "Legal")
                sections = doc_data.get("sections", [])
                
                for sec in sections:
                    sec_id = sec.get("section_id", "Sec")
                    sec_title = sec.get("title", "Provision")
                    sec_text = sec.get("text", "").strip()
                    
                    if not sec_text:
                        continue
                        
                    chunk_id = f"bge_doc_{track_name}_{doc_data.get('doc_id', 'doc')}_{sec_id}"
                    full_content = f"Statute: {act_name}\nProvision: {sec_title}\nCategory: {category}\nContent: {sec_text}"
                    w_cnt = len(full_content.split())
                    
                    meta = {
                        "chunk_id": chunk_id,
                        "source_type": "statute_document",
                        "track": track_name,
                        "act_name": act_name,
                        "category": category,
                        "section_id": sec_id,
                        "title": sec_title,
                        "word_count": int(w_cnt)
                    }
                    
                    if chunk_id not in all_ids:
                        all_ids.append(chunk_id)
                        all_documents.append(full_content)
                        all_metadatas.append(meta)
                        processed_docs_count += 1
                    
            except Exception as de:
                print(f"Failed loading document {df_name}: {de}", flush=True)

    # 2. Transcripts
    trans_dir = paths["transcripts"]
    if os.path.exists(trans_dir):
        trans_files = [f for f in os.listdir(trans_dir) if f.endswith(".json") and not f.endswith(".partial.json")]
        for tf_name in trans_files:
            tf_path = os.path.join(trans_dir, tf_name)
            try:
                with open(tf_path, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                    
                vid = t_data.get("video_id", tf_name.replace(".json", ""))
                c_name = t_data.get("case_name", vid)
                segments = t_data.get("segments", [])
                
                t_chunks = chunk_transcript_dialogues_bge(
                    video_id=vid,
                    case_name=c_name,
                    track_name=track_name,
                    segments=segments,
                    target_word_min=150,
                    target_word_max=300
                )
                
                for tc in t_chunks:
                    all_ids.append(tc["id"])
                    all_documents.append(tc["text"])
                    all_metadatas.append(tc["metadata"])
                    processed_speech_chunks += 1
                    
            except Exception as te:
                print(f"Failed loading transcript {tf_name}: {te}", flush=True)

total_chunks = len(all_ids)
print(f"Total BGE Chunks to Index: {total_chunks} (Speech: {processed_speech_chunks}, Docs: {processed_docs_count})", flush=True)

# ──────────────────────────────────────────────────────────────────────
# Fast Batch Embedding & Vector Store Upsert (Batch Size = 256)
# ──────────────────────────────────────────────────────────────────────
batch_size = 256

print(f"\nAccelerated BGE-Large batch embeddings ({total_chunks} total chunks)...", flush=True)

for i in range(0, total_chunks, batch_size):
    b_ids = all_ids[i : i + batch_size]
    b_docs = all_documents[i : i + batch_size]
    b_metas = all_metadatas[i : i + batch_size]
    
    b_embs = embedder.encode(b_docs, batch_size=32, show_progress_bar=False).tolist()
    
    collection.upsert(
        ids=b_ids,
        documents=b_docs,
        metadatas=b_metas,
        embeddings=b_embs
    )
    print(f"  Progress: [{min(i + batch_size, total_chunks)} / {total_chunks}] chunks embedded and indexed into ChromaDB.", flush=True)

print(f"\n[SUCCESS] RE-INDEXING WITH BGE COMPLETE!", flush=True)
print(f"Total Chunks in 'legal_speech_rag_bge': {collection.count()}", flush=True)
