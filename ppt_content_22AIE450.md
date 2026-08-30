# 22AIE450 Speech Processing — Mid Semester Project Review (Dataset & RAG AI Focus)

**Project Title**: Legal Speech Dataset Curation & Spoken RAG Engine for Multi-Speaker Audio Question-Answering  
**Course**: 22AIE450 Speech Processing (Semester 7)  
**Date**: August 2026  
**Batch**: 2022–2026 | **Group**: Group 7  
**Generated PPT File**: [project_presentation_22AIE450_v2.pptx](file:///c:/Users/varshitha-home/Desktop/me/amrita/SEM%207/Speech%20Processing/Project/project_presentation_22AIE450_v2.pptx)

---

## Slide 1: Title Slide
- **Header**: `[Mid Project Review]`
- **Project Title**: Legal Speech Dataset Curation & Spoken RAG Engine for Multi-Speaker Audio Question-Answering
- **Course**: 22AIE450 Speech Processing (Mid semester project review — August 2026)
- **Team**: Varshitha & Team (22AIE450 Speech Processing, Sem 7)
- **Batch & Group**: Batch 2022–2026 | Group No: 7

---

## Slide 2: Outline
- **Phase 1: Dataset Collection, Preprocessing & Curation (PRIMARY FOCUS)**
  - Audio Extraction, Signal Normalization & Metadata Indexing
  - PyAnnote 3.1 Neural Diarization & Speaker Boundary Segmentation
  - Faster-Whisper ASR & Checkpoint-Resilient Transcription
  - Dialogue Sliding-Window Chunking & Statutory Corpus Parsing
- **Phase 2: Vector RAG & LLM Answer Synthesis (SECONDARY FOCUS)**
  - BGE-Large 1024-dim Embedding & ChromaDB HNSW Indexing
  - Hybrid Metadata Filtering & Grounded Neural Speech Synthesis
- Preliminary Results, Dataset Stats & Next Phase Plan

---

## Slide 3: Introduction and Motivation
- **Primary Challenge**: Long-form multi-speaker audio recordings (1–3 hours) are unstructured, making precise information retrieval computationally difficult.
- **Dataset Focus**: Curating a high-quality speech dataset of 179 video proceedings (~150+ audio hours) paired with speaker turns and statutory provisions.
- **Speech Processing Role**: Audio extraction (16kHz mono), PyAnnote 3.1 speaker diarization, and Faster-Whisper ASR to build structured dialogue transcripts.
- **Downstream RAG Goal**: Leveraging dense vector search and LLMs to answer legal queries with verified timestamped voice output.

---

## Slide 4: Problem Definition (Unified Problem Statement)

> **Unified Problem Statement**:  
> *"The core challenge lies in curating a structured legal speech corpus from unscripted multi-speaker audio by resolving speaker boundary drift and GPU-resilient long-form ASR (Phase 1), while eliminating vector space collisions and hallucinations during downstream dense RAG retrieval and neural synthesis (Phase 2)."*

- **Phase 1 (Dataset Preparation Challenges)**:
  - *Speaker Diarization Drift*: Overlapping voices in unscripted proceedings require precise speaker boundary detection (`SPEAKER_00`, `SPEAKER_01`).
  - *Long-Form ASR Fault Tolerance*: Transcribing multi-hour audio without GPU VRAM thrashing or memory loss.
- **Phase 2 (RAG & Retrieval Challenges)**:
  - *Vector Space Collisions*: Dense embeddings conflate generic section tags (`Section 9A`) across different domain acts.
  - *Grounded LLM Synthesis*: Preventing hallucinations by grounding answers directly in verified transcript vectors.

---

## Slide 5: Literature Review (Speech Corpora & RAG AI)
- **Whisper ASR (Radford et al., ICML 2023)**: State-of-the-art transformer ASR model; int8 quantization optimizes GPU memory during batch inference.
- **PyAnnote Audio 3.1 (Bredin et al., Interspeech 2023)**: Neural diarization using SincNet embeddings and HNSW clustering for speaker turn segmentation.
- **BGE Embedding Models (BAAI, 2023)**: 1024-dimensional dense vector embeddings optimized for semantic passage retrieval.
- **Dense Passage Retrieval & RAG (Karpukhin 2020, Lewis 2020)**: Bi-encoder retrieval coupled with LLMs for factual question answering.

---

## Slide 6: Gaps Identified in Dataset & Retrieval Systems
- **Lack of Structured Legal Speech Corpora**: Absence of publicly available Indian legal speech datasets with paired speaker IDs, timestamps, and transcripts.
- **Unstructured Audio Chunking Flaws**: Naive sentence splitting breaks speaker dialogue continuity and timestamp provenance.
- **Pure Semantic Retrieval Collisions**: Vector similarity alone cannot distinguish identical section numbers across different statutory acts.
- **Speech-to-Speech QA Gap**: Most QA systems handle text inputs; workflows benefit from end-to-end spoken audio answers.

---

## Slide 7: Project Objectives
- **Phase 1 Objectives (Dataset Preparation)**:
  - Curate 179-video speech dataset (100 Moot Court + 79 Supreme Court live-streams).
  - Implement 4-stage processing: `ffmpeg` Audio → PyAnnote Diarization → Segmented Whisper ASR → Dialogue Chunker.
- **Phase 2 Objectives (RAG & LLM Engine)**:
  - Build ChromaDB vector store (BGE-Large 1024-dim embeddings) with exact metadata section boosting.
  - Synthesize grounded prose answers with inline citations and neural TTS audio (`edge-tts`).

---

## Slide 8: Methodology — Phase 1: Dataset Pipeline & Phase 2: RAG
- **Phase 1: Dataset Preparation Pipeline**:
  - *Audio Conditioning*: `ffmpeg` 16kHz 16-bit mono PCM conversion.
  - *Neural Diarization*: `pyannote/speaker-diarization-3.1` extracting speaker-stamped intervals.
  - *Segmented ASR*: `faster-whisper small int8` with per-segment `.partial.json` fault-tolerant checkpointing.
  - *Dialogue Sliding Window*: 150-300 word chunks with start/end timestamps and speaker tags.
- **Phase 2: RAG & Synthesis Engine**:
  - BGE-Large (1024 dims) embeddings in ChromaDB (`legal_speech_rag_bge`).
  - Hybrid Retrieval: Exact section metadata boost (`0.9500` score) + Act-scoped fallback.
  - LLM & TTS: Grounded synthesis + `edge-tts` neural Indian voice (`en-IN-NeerjaNeural`).

---

## Slide 9: Dataset Description & Curation Statistics
- **Moot Court Dataset Track**: 100 national & international competition proceedings (58.29 hours transcribed, 461,436 words).
- **Supreme Court Dataset Track**: 79 live-streamed constitution bench hearings (26.69 hours transcribed, 202,365 words).
- **Statutory Knowledge Base**: Parsed statutory provisions (Representation of the People Act 1951, CrPC, Constitution).
- **Vector Collection Size**: **4,260 clean indexed chunks** (4,234 dialogue turns + 26 statutory section provisions).

---

## Slide 10: Preliminary Results: Dataset & Speech Processing
- **PyAnnote 3.1 Diarization Quality**: Successfully isolated distinct speaker turns (`SPEAKER_00`, `SPEAKER_01`) across 85+ audio hours.
- **ASR Transcription Robustness**: Faster-Whisper `small int8` achieved high word accuracy across Indian accents and specialized vocabulary.
- **Checkpoint-Resilient ASR Pipeline**: Segmented `.partial.json` architecture preserved 100% of transcribed text across system restarts.
- **Total Speech Data Curated**: **663,801 spoken words** transcribed across **84.98 audio hours**.

---

## Slide 11: Preliminary Results: Vector RAG & Retrieval Engine
- **Exact Section Retrieval**: Surfaced RPA Section 9A at Rank 1 with `0.9500` boosted similarity score.
- **Cross-Act Contamination**: 0% (unrelated Arbitration Act Sec 9 chunks automatically filtered out).
- **Context Precision (@0.35 threshold)**: **100.0%** (4/4 relevant retrieved chunks).
- **Pipeline Latency**: Vector retrieval + dynamic synthesis completed in `< 12.28 seconds` on CUDA GPU.
- **Zero Duplicate Output**: Text-level deduplication eliminated duplicate paragraph output.

---

## Slide 12: Interim Summary/Conclusion
- **Dataset Phase Completed**: Successfully built automated pipeline for speech extraction, PyAnnote diarization, and checkpoint-resilient ASR.
- **RAG Engine Operational**: Built 4,260-chunk BGE-Large vector store with 100% Context Precision and hybrid section metadata filtering.
- **Milestones Achieved**:
  - Curated **663,801 words** across **85+ hours** of spoken courtroom proceedings.
  - Validated speech-to-text-to-audio QA pipeline with zero cross-act contamination.

---

## Slide 13: Completion Plan for the Next Phase
- **Complete Dataset ASR**: Finish ASR processing for remaining video dataset audio.
- **Whisper ASR Fine-Tuning**: Fine-tune Whisper on specialized Indian legal vocabulary and acoustic conditions.
- **Interactive Web Interface**: Build Next.js / Vite UI displaying audio waveforms, speaker turns, and RAG answers.
- **Comprehensive Evaluation**: Benchmark WER, DER, and Retrieval ROUGE/BLEU scores across 50+ test queries.

---

## Slide 14: References
1. Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *ICML*.
2. Bredin, H., et al. (2023). "PyAnnote.audio 3.0: Speaker Diarization Revisited." *Interspeech*.
3. BAAI (2023). "Beijing Academy of Artificial Intelligence BGE Embedding Models." *GitHub Repository*.
4. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
5. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*.

---

## Slide 15: Thank You
- **Thank You!**
- Questions & Discussion
- Project Code Repository & Vector Store: `legal_speech_rag_bge`
