import os
import sys
import json
import logging
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Setup Paths for Vector Database & Dataset Tracks
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(".")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "legal_rag_vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

TRACKS = {
    "moot_court": {
        "transcripts": os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "transcripts"),
        "documents":   os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "documents"),
        "metadata":    os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "metadata", "video_metadata.csv"),
    },
    "supreme_court": {
        "transcripts": os.path.join(PROJECT_ROOT, "supreme court", "transcripts"),
        "documents":   os.path.join(PROJECT_ROOT, "supreme court", "documents"),
        "metadata":    os.path.join(PROJECT_ROOT, "supreme court", "metadata", "video_metadata.csv"),
    }
}

logger = logging.getLogger("stage4_indexer")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)

logger.info("Initializing Stage 4 — RAG Vector Indexer...")

# ──────────────────────────────────────────────────────────────────────
# Load Embedding Model & ChromaDB Client
# ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading SentenceTransformer embedding model '{EMBEDDING_MODEL_NAME}'...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded successfully!")

chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = chroma_client.get_or_create_collection(
    name="legal_speech_rag",
    metadata={"hnsw:space": "cosine"}
)

print(f"ChromaDB Collection initialized at: {VECTORSTORE_DIR}")

# ──────────────────────────────────────────────────────────────────────
# Helper: Create Sliding Window Chunks for Transcripts
# ──────────────────────────────────────────────────────────────────────
def chunk_transcript_dialogues(video_id, case_name, track_name, segments, window_size=3, step=2):
    """
    Groups consecutive dialogue turns into sliding window chunks to preserve
    conversational context between judges and advocates.
    """
    chunks = []
    if not segments:
        return chunks

    for i in range(0, len(segments), step):
        window = segments[i : i + window_size]
        if not window:
            continue
        
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
        if len(combined_text.strip()) < 15:
            continue
            
        chunk_id = f"{track_name}_{video_id}_chunk_{i}"
        metadata = {
            "chunk_id": chunk_id,
            "source_type": "speech_transcript",
            "track": track_name,
            "video_id": str(video_id),
            "case_name": str(case_name)[:100],
            "start_time": float(start_time),
            "end_time": float(end_time),
            "speakers": ", ".join(speakers),
            "num_turns": len(window)
        }
        
        chunks.append({
            "id": chunk_id,
            "text": f"Case: {case_name}\nTrack: {track_name}\n" + combined_text,
            "metadata": metadata
        })

    return chunks

# ──────────────────────────────────────────────────────────────────────
# Ingest Data from Both Tracks
# ──────────────────────────────────────────────────────────────────────
all_documents = []
all_metadatas = []
all_ids = []

processed_docs_count = 0
processed_speech_chunks = 0

for track_name, paths in TRACKS.items():
    logger.info(f"\nProcessing Track: {track_name}")
    
    # 1. Ingest Statutory Documents
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
                        
                    chunk_id = f"doc_{track_name}_{doc_data.get('doc_id', 'doc')}_{sec_id}"
                    full_content = f"Statute: {act_name}\nProvision: {sec_title}\nCategory: {category}\nContent: {sec_text}"
                    
                    meta = {
                        "chunk_id": chunk_id,
                        "source_type": "statute_document",
                        "track": track_name,
                        "act_name": act_name,
                        "category": category,
                        "section_id": sec_id,
                        "title": sec_title
                    }
                    
                    all_ids.append(chunk_id)
                    all_documents.append(full_content)
                    all_metadatas.append(meta)
                    processed_docs_count += 1
                    
            except Exception as de:
                logger.warning(f"Failed loading document {df_name}: {de}")

    # 2. Ingest Transcripts
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
                
                t_chunks = chunk_transcript_dialogues(
                    video_id=vid,
                    case_name=c_name,
                    track_name=track_name,
                    segments=segments,
                    window_size=3,
                    step=2
                )
                
                for tc in t_chunks:
                    all_ids.append(tc["id"])
                    all_documents.append(tc["text"])
                    all_metadatas.append(tc["metadata"])
                    processed_speech_chunks += 1
                    
            except Exception as te:
                logger.warning(f"Failed loading transcript {tf_name}: {te}")

logger.info(f"\nTotal Chunk Elements to Index:")
logger.info(f"  - Statutory Document Chunks: {processed_docs_count}")
logger.info(f"  - Spoken Transcript Chunks: {processed_speech_chunks}")
logger.info(f"  - Combined Total Chunks   : {len(all_ids)}")

# ──────────────────────────────────────────────────────────────────────
# Batch Embedding & Vector Store Upsert
# ──────────────────────────────────────────────────────────────────────
if all_documents:
    print(f"\nGenerating embeddings for {len(all_documents)} chunks using {EMBEDDING_MODEL_NAME}...")
    embeddings = embedder.encode(all_documents, batch_size=64, show_progress_bar=True).tolist()
    
    print("Upserting chunks and vectors into ChromaDB collection...")
    # Upsert in batches of 500 for memory efficiency
    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        b_ids = all_ids[i : i + batch_size]
        b_docs = all_documents[i : i + batch_size]
        b_metas = all_metadatas[i : i + batch_size]
        b_embs = embeddings[i : i + batch_size]
        
        collection.upsert(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_metas,
            embeddings=b_embs
        )
        print(f"  Indexed batch [{i + len(b_ids)} / {len(all_ids)}]")

print(f"\n[SUCCESS] STAGE 4 VECTOR INDEXING COMPLETE!")
print(f"Total Chunks Stored in ChromaDB: {collection.count()}")
print(f"Vector Database Location: {VECTORSTORE_DIR}")
