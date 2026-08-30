import os
import sys
import time
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Import RAG pipeline from legal_rag_qa
try:
    from legal_rag_qa import (
        process_hierarchical_legal_query,
        collection,
        LEGAL_DOMAINS,
        SUBDOMAIN_APPLICABLE_LAW,
        EMBEDDING_MODEL_NAME,
        OUTPUT_AUDIO_DIR
    )
except Exception as e:
    logging.error(f"Error importing legal_rag_qa modules: {e}")
    raise e

logger = logging.getLogger("legal_rag_api")
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────
# FastAPI App Initialization
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal Speech RAG API",
    description="Production-grade REST API for Indian Legal Speech Question Answering and Spoken Neural Synthesis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for worldwide frontend clients (Vercel, Localhost, Mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────
# Pydantic Request & Response Models
# ──────────────────────────────────────────────────────────────────────
class LegalQueryRequest(BaseModel):
    query_text: str = Field(..., example="What does Section 9A of the Representation of the People Act state regarding disqualification for government contracts?")
    domain: str = Field(..., example="Constitutional & Administrative Law")
    subdomain: str = Field(..., example="Election & Representation Law")
    voice_gender: Optional[str] = Field("Female", example="Female")

class EvaluationMetrics(BaseModel):
    mean_cosine_similarity: float
    max_cosine_similarity: float
    context_precision: float
    principled_threshold: float
    retrieved_chunks_count: int
    relevant_chunks_count: int
    exact_match_chunks: int
    semantic_match_chunks: int
    response_word_count: int
    response_char_count: int
    total_latency_seconds: float

class LegalQueryResponse(BaseModel):
    success: bool
    query: str
    domain: str
    subdomain: str
    applicable_law: str
    answer: str
    citations: str
    audio_url: Optional[str] = None
    audio_filename: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# ──────────────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for cloud monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "indexed_chunks": collection.count() if collection else 0
    }

@app.get("/api/domains", tags=["Legal Knowledge Base"])
async def get_legal_domains():
    """Returns the complete legal domain and subdomain hierarchy along with governing statutory laws."""
    domains_data = []
    for domain_name, sub_list in LEGAL_DOMAINS.items():
        subdomain_items = []
        for s in sub_list:
            subdomain_items.append({
                "subdomain_name": s,
                "applicable_law": SUBDOMAIN_APPLICABLE_LAW.get(s, f"{s} (Applicable Indian Statute)")
            })
        domains_data.append({
            "domain_name": domain_name,
            "subdomains": subdomain_items
        })
    return {"domains": domains_data}

@app.get("/api/stats", tags=["Legal Knowledge Base"])
async def get_knowledge_base_stats():
    """Returns live collection metrics, statutory chunk count, and transcript chunk count."""
    try:
        total_chunks = collection.count()
        all_data = collection.get(include=["metadatas"])
        metas = all_data.get("metadatas", [])
        
        statute_count = sum(1 for m in metas if m.get("source_type") == "statute_document")
        speech_count = sum(1 for m in metas if m.get("source_type") == "speech_transcript")
        
        return {
            "total_chunks_indexed": total_chunks,
            "statute_provisions_count": statute_count,
            "moot_court_speech_turns": speech_count,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "jurisdiction": "India"
        }
    except Exception as e:
        logger.error(f"Error reading vectorstore stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=LegalQueryResponse, tags=["RAG QA Engine"])
async def answer_legal_query(payload: LegalQueryRequest):
    """
    Executes domain guardrails, exact section boosting, vector search,
    21-rule legal synthesis, and neural Indian voice generation.
    """
    try:
        t0 = time.time()
        logger.info(f"Processing query: '{payload.query_text}' under {payload.domain} -> {payload.subdomain}")
        
        answer, citations, audio_path, metrics = process_hierarchical_legal_query(
            query_text=payload.query_text,
            domain=payload.domain,
            subdomain=payload.subdomain,
            voice_gender=payload.voice_gender or "Female"
        )
        
        audio_filename = os.path.basename(audio_path) if audio_path and os.path.exists(audio_path) else None
        audio_url = f"/api/audio/{audio_filename}" if audio_filename else None
        
        applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(payload.subdomain, payload.subdomain)
        
        return LegalQueryResponse(
            success=True,
            query=payload.query_text,
            domain=payload.domain,
            subdomain=payload.subdomain,
            applicable_law=applicable_law,
            answer=answer,
            citations=citations,
            audio_url=audio_url,
            audio_filename=audio_filename,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.get("/api/audio/{filename}", tags=["Audio Engine"])
async def stream_audio_file(filename: str):
    """Streams generated neural MP3 audio file directly to browser/frontend player."""
    file_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or expired")
    
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )

# ──────────────────────────────────────────────────────────────────────
# Entry point for direct execution
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
