---
title: Legal Speech API
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.15.0
app_file: app_hf.py
pinned: false
license: mit
short_description: Multi-speaker legal audio transcript RAG & QA Engine
---

# ⚖️ Legal Speech RAG Studio
### Multi-Speaker Audio Question-Answering & Statutory Verification Engine

An end-to-end Spoken RAG (Retrieval-Augmented Generation) system built for multi-speaker Indian legal proceedings and statutory knowledge verification.

---

## 🌟 Key Features
- **Dense Vector Search**: ChromaDB HNSW indexing with 1024-dimensional `BAAI/bge-large-en-v1.5` embeddings.
- **Exact Metadata Section Boosting**: Instant priority matching for statutory sections (e.g. RPA Section 9A, Arbitration Act Sec 34) with zero cross-act contamination.
- **Audio Transcript Grounding**: 4,260 clean dialogue chunks curated from 85+ audio hours with PyAnnote 3.1 neural speaker diarization and Faster-Whisper ASR.
- **Neural Speech Synthesis**: Generates audible legal answers via `edge-tts` (`en-IN-NeerjaNeural`).
- **Interactive Streamlit Interface**: Dual-page dashboard for legal QA query exploration and live audio transcription & summarization.

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-dir>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
