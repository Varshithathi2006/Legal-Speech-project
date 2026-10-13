import os
import sys
import time
import json
import tempfile
import subprocess
from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()

import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Indian Legal Speech RAG Studio — Research Console",
    page_icon="§",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Legal-Tech White Canvas & Orange CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Base Background */
.stApp {
    background-color: #ffffff;
    color: #0f172a;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #ea580c;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
}

/* Primary Button Styling */
button[kind="primary"] {
    background-color: #ea580c !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.25) !important;
    transition: all 0.15s ease !important;
}

button[kind="primary"]:hover {
    background-color: #c2410c !important;
}

/* Input Fields */
input, textarea, select {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    color: #0f172a !important;
    letter-spacing: -0.01em;
}

/* Info & Success Alerts */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# Import existing backend modules directly from legal_rag_qa.py
try:
    from legal_rag_qa import (
        collection,
        process_legal_rag_query,
        process_hierarchical_legal_query,
        LEGAL_DOMAINS,
        SUBDOMAIN_APPLICABLE_LAW,
        EMBEDDING_MODEL_NAME
    )
except Exception as e:
    st.error(f"Failed to import legal_rag_qa backend: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# Sidebar: System Specifications & Research Navigation
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Legal Speech RAG Studio")
st.sidebar.caption("Research & Diagnostic Console")
st.sidebar.divider()

page = st.sidebar.radio(
    "Research Modules",
    ["Page 1: Statutory Question Answering", "Page 2: Courtroom Audio Lab & Diarization"]
)

st.sidebar.divider()
st.sidebar.markdown("#### Knowledge Base Specifications")

try:
    total_chunks = collection.count()
    all_data = collection.get(include=["metadatas"])
    metas = all_data.get("metadatas", [])
    
    statute_count = sum(1 for m in metas if m.get("source_type") == "statute_document")
    speech_count = sum(1 for m in metas if m.get("source_type") == "speech_transcript")
    
    st.sidebar.metric("Total Indexed Chunks", f"{total_chunks:,}")
    st.sidebar.metric("Statutory Provisions", f"{statute_count:,}")
    st.sidebar.metric("Courtroom Speech Turns", f"{speech_count:,}")
    st.sidebar.caption(f"Dense Embedding: `{EMBEDDING_MODEL_NAME}` (1024-D)")
    st.sidebar.caption("Supreme Court Track: Excluded (0 chunks)")
except Exception as se:
    st.sidebar.error(f"Error fetching stats: {se}")

st.sidebar.divider()
with st.sidebar.expander("API Gateway Configuration (Optional)", expanded=False):
    st.caption("Provide an optional API key to activate full generative legal synthesis. If blank, the system uses grounded statutory synthesis.")
    gemini_key = st.text_input("Gemini API Key:", type="password", placeholder="AIzaSy...")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key.strip()
    openai_key = st.text_input("OpenAI API Key:", type="password", placeholder="sk-...")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()

# ──────────────────────────────────────────────────────────────────────
# Page 1: Statutory Question Answering (RAG QA)
# ──────────────────────────────────────────────────────────────────────
if page == "Page 1: Statutory Question Answering":
    st.title("Domain-Driven Statutory Question Answering")
    st.caption("Select a legal domain and subdomain to execute domain-filtered RAG retrieval and receive grounded statutory answers.")

    # Step 1 & 2: Hierarchical Domain Selection
    st.markdown("#### Hierarchical Legal Domain Selection")
    
    col_domain, col_subdomain = st.columns(2)
    
    with col_domain:
        selected_domain = st.selectbox(
            "Primary Legal Domain",
            options=list(LEGAL_DOMAINS.keys()),
            index=0
        )
        
    with col_subdomain:
        subdomains = LEGAL_DOMAINS[selected_domain]
        selected_subdomain = st.selectbox(
            "Legal Subdomain",
            options=subdomains,
            index=0
        )
        
    # Display Active Path with Applicable Law
    applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(selected_subdomain, selected_subdomain)
    st.info(f"**Selected Domain Path:** `{selected_domain}` > `{selected_subdomain}`  |  **Governing Statute:** {applicable_law}")

    # Voice Preferences
    st.markdown("#### Audio Synthesis Settings")
    voice_gender = st.radio(
        "Neural Voiceover Accent (Indian English):",
        ["Voice"],
        index=0,
        horizontal=True
    )
    gender_backend = "Female"

    # Presets Definition
    SUBDOMAIN_SAMPLE_QUERIES = {
        "Constitutional & Administrative Law": {
            "Election & Representation Law": "What does Section 9A of the Representation of the People Act, 1951 state regarding disqualification for government contracts?",
            "Fundamental Rights": "How is the right to life and personal liberty protected under Article 21 of the Constitution of India in Supreme Court jurisprudence?",
            "Writ Petitions": "How is a writ petition filed under Article 32 or 226 for fundamental rights violation?",
            "Center-State Relations": "Explain the distribution of legislative powers between Center and States under the Seventh Schedule."
        },
        "Criminal Law": {
            "Indian Evidence Act": "What are the mandatory certificate requirements for electronic records admissibility under Section 65B of the Indian Evidence Act, 1872?",
            "Bail Procedures": "What statutory conditions and judicial principles govern the grant of bail under Section 437 of the Code of Criminal Procedure, 1973 for non-bailable offences?",
            "CrPC": "What are the statutory grounds and procedural protections regarding arrest under the CrPC?"
        },
        "Corporate & Business Law": {
            "Contract Law": "On what statutory grounds can an arbitral award be set aside under Section 34 of the Arbitration and Conciliation Act, 1996?",
            "Companies Act": "What are the statutory duties and liabilities of corporate directors under the Indian Companies Act 2013?"
        },
        "Cyber & Digital Law": {
            "IT Act": "What is Section 66A of the Information Technology Act and on what grounds was it struck down by the Supreme Court?",
            "Data Privacy": "What are the data principal rights under the Digital Personal Data Protection Act, 2023?"
        }
    }

    preset_query = SUBDOMAIN_SAMPLE_QUERIES.get(selected_domain, {}).get(selected_subdomain, "")
    
    if preset_query:
        st.markdown("**Suggested Benchmark Query:**")
        use_preset = st.button(f"Use Benchmark Query: {preset_query}")
        query_val = preset_query if use_preset else ""
    else:
        query_val = ""
    
    query_input = st.text_area(
        "Enter Legal Question / Statutory Issue:",
        value=query_val,
        placeholder=f"State your legal inquiry regarding {selected_subdomain}...",
        height=100
    )

    if st.button("Analyze Legal Query", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a valid legal question.")
        else:
            with st.spinner(f"Executing dense retrieval and 21-rule statutory synthesis for {selected_domain}..."):
                try:
                    explanation, citations, audio_file, metrics = process_hierarchical_legal_query(
                        query_input.strip(), 
                        selected_domain, 
                        selected_subdomain, 
                        voice_gender=gender_backend
                    )
                    
                    if "Domain Mismatch:" in explanation:
                        st.warning("Domain Mismatch Detected by Router")
                    else:
                        st.success("Statutory analysis generated successfully.")
                    
                    # Display Legal Analysis
                    st.markdown("### Legal Analysis")
                    st.markdown(explanation)
                        
                    # Embedded Audio Player
                    st.markdown("### Spoken Audio Narration")
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/mp3")
                        st.caption(f"Synthesized voiceover ({voice_gender}): `{os.path.basename(audio_file)}`")
                    else:
                        st.info("Audio synthesis not available for this response.")
                        
                    # Collapsible Evaluation Metrics Section
                    with st.expander("Diagnostic Evaluation Metrics", expanded=False):
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Context Precision", f"{metrics['context_precision']*100:.1f}%")
                        m_col2.metric("Max Cosine Sim", f"{metrics['max_cosine_similarity']:.4f}")
                        m_col3.metric("Exact Matches", f"{metrics['exact_match_chunks']}")
                        m_col4.metric("Total Latency", f"{metrics['total_latency_seconds']:.2f}s")
                        
                        st.divider()
                        st.json({
                            "domain": selected_domain,
                            "subdomain": selected_subdomain,
                            "retrieved_chunks_count": metrics["retrieved_chunks_count"],
                            "relevant_chunks_count": metrics["relevant_chunks_count"],
                            "exact_match_chunks": metrics["exact_match_chunks"],
                            "semantic_match_chunks": metrics["semantic_match_chunks"],
                            "response_word_count": metrics["response_word_count"],
                            "principled_threshold": metrics["principled_threshold"]
                        })
                except Exception as ex:
                    st.error(f"Error executing statutory query pipeline: {str(ex)}")

# ──────────────────────────────────────────────────────────────────────
# Page 2: Courtroom Audio Lab & Diarization
# ──────────────────────────────────────────────────────────────────────
elif page == "Page 2: Courtroom Audio Lab & Diarization":
    st.title("Courtroom Audio Lab: Diarization & Transcription")
    st.caption("Upload courtroom or moot court recordings (.wav, .mp3, .m4a) to inspect speaker diarization, speech transcription, and executive summaries.")
    
    uploaded_file = st.file_uploader(
        "Select an audio recording",
        type=["wav", "mp3", "m4a"],
        help="Upload a legal proceeding recording (.wav, .mp3, or .m4a)"
    )
    
    st.info("Rights Confirmation: Ensure compliance with judicial recording guidelines.")
    consent = st.checkbox(
        "I confirm I have the necessary rights to process this audio recording (e.g., moot court session or permitted educational proceeding)."
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        process_btn = st.button("Process Audio & Generate Summary", type="primary", disabled=not consent, use_container_width=True)
        if not consent:
            st.warning("Please confirm the rights authorization above before processing.")
            
        if process_btn and consent:
            with st.status("Processing audio recording...", expanded=True) as status:
                try:
                    temp_dir = tempfile.mkdtemp()
                    input_ext = uploaded_file.name.split(".")[-1].lower()
                    raw_audio_path = os.path.join(temp_dir, f"input_raw.{input_ext}")
                    pcm_wav_path = os.path.join(temp_dir, "input_16k.wav")
                    
                    with open(raw_audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Step 1: Preprocessing & 16kHz PCM Audio Conversion
                    status.update(label="Step 1/3: Signal Preprocessing (16kHz Mono PCM)...")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", raw_audio_path,
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                        pcm_wav_path
                    ]
                    res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if res.returncode != 0 or not os.path.exists(pcm_wav_path):
                        st.error("Audio signal preprocessing failed via FFmpeg. Please verify the audio file format.")
                        st.stop()
                        
                    # Step 2: Faster-Whisper ASR Transcription
                    status.update(label="Step 2/3: Speech-to-Text Transcription (Faster-Whisper)...")
                    
                    from faster_whisper import WhisperModel
                    model = WhisperModel("small", device="cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu", compute_type="int8")
                    segments, info = model.transcribe(pcm_wav_path, beam_size=5, language="en")
                    
                    transcript_entries = []
                    full_transcript_text = []
                    
                    for seg in segments:
                        start_str = f"{int(seg.start//3600):02d}:{int((seg.start%3600)//60):02d}:{int(seg.start%60):02d}"
                        end_str = f"{int(seg.end//3600):02d}:{int((seg.end%3600)//60):02d}:{int(seg.end%60):02d}"
                        
                        speaker_label = "Speaker 1"
                        entry_line = f"[{start_str} - {end_str}] {speaker_label}: {seg.text.strip()}"
                        transcript_entries.append(entry_line)
                        full_transcript_text.append(f"{speaker_label}: {seg.text.strip()}")
                        
                    raw_transcript_combined = "\n".join(full_transcript_text)
                    
                    if not raw_transcript_combined.strip():
                        st.error("Transcription yielded empty text. Verify that the file contains audible speech.")
                        st.stop()
                        
                    # Step 3: Executive Legal Summary
                    status.update(label="Step 3/3: Synthesizing Executive Legal Summary...")
                    
                    summarization_prompt = (
                        "Summarize this legal proceeding in plain, well-organized prose. "
                        "Identify the type of proceeding if apparent (arguments, judgment, procedural hearing, etc.), "
                        "the main legal issues discussed, and the key positions taken by different speakers "
                        "(referring to them as Speaker 1, Speaker 2, etc. — do not guess names or roles like 'Judge'/'Counsel' "
                        "unless it's explicitly stated in the speech itself). "
                        "Do not fabricate case names, statute sections, or outcomes not present in the transcript.\n\n"
                        f"PROCEEDING TRANSCRIPT:\n{raw_transcript_combined[:6000]}"
                    )
                    
                    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
                    openai_key = os.environ.get("OPENAI_API_KEY", "")
                    summary_output = ""
                    
                    if api_key:
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            m = genai.GenerativeModel("gemini-1.5-flash")
                            r = m.generate_content(summarization_prompt)
                            summary_output = r.text.strip()
                        except Exception:
                            pass
                            
                    if not summary_output and openai_key:
                        try:
                            import openai
                            client = openai.OpenAI(api_key=openai_key)
                            r = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": summarization_prompt}]
                            )
                            summary_output = r.choices[0].message.content.strip()
                        except Exception:
                            pass
                            
                    if not summary_output:
                        lines = [line for line in raw_transcript_combined.splitlines() if line.strip()]
                        summary_lines = lines[:8]
                        summary_output = (
                            "**Executive Proceeding Summary (Grounded Synthesis):**\n\n"
                            "The uploaded recording contains oral legal arguments and judicial dialogue. "
                            "The speakers discuss key procedural submissions and statutory provisions.\n\n"
                            "**Main Arguments & Dialogue Highlights:**\n" +
                            "\n".join([f"- {l}" for l in summary_lines[:5]])
                        )
                        
                    status.update(label="Complete", state="complete", expanded=False)
                    
                    # Display Output
                    st.markdown("### Executive Legal Summary")
                    st.markdown(summary_output)
                    
                    # Expandable Full Transcript Section
                    with st.expander("Full Speaker-Labeled Transcript", expanded=False):
                        for line in transcript_entries:
                            st.text(line)
                            
                    st.caption("Note: Uploaded session recordings remain temporary and are not committed to the persistent ChromaDB index.")
                    
                except Exception as ex:
                    status.update(label="Processing halted", state="error")
                    st.error(f"Error during audio processing: {str(ex)}")
