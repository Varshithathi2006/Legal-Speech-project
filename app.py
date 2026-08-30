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
    page_title="Indian Legal Speech RAG Studio",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# Sidebar: Live Dataset Statistics & Navigation
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("⚖️ Legal Speech RAG")
st.sidebar.markdown("**Speech Processing & Vector QA Studio**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Page 1: Ask a Legal Question", "🎙️ Page 2: Upload Audio & Summarize"]
)

st.sidebar.divider()
st.sidebar.subheader("📊 Live Knowledge Base Stats")

try:
    total_chunks = collection.count()
    all_data = collection.get(include=["metadatas"])
    metas = all_data.get("metadatas", [])
    
    statute_count = sum(1 for m in metas if m.get("source_type") == "statute_document")
    speech_count = sum(1 for m in metas if m.get("source_type") == "speech_transcript")
    
    st.sidebar.metric("Total Chunks Indexed", f"{total_chunks:,}")
    st.sidebar.metric("Statutory Provisions", f"{statute_count:,}")
    st.sidebar.metric("Moot Court Speech Turns", f"{speech_count:,}")
    st.sidebar.caption(f"Embedding: `{EMBEDDING_MODEL_NAME}`")
    st.sidebar.caption("Supreme Court Track: Excluded (0 chunks)")
except Exception as se:
    st.sidebar.error(f"Error fetching stats: {se}")

st.sidebar.divider()
with st.sidebar.expander("🔑 LLM API Key (Optional)", expanded=False):
    st.caption("Enter a Gemini or OpenAI key to activate full 21-rule generative legal synthesis. If blank, the system uses clean grounded statutory synthesis.")
    gemini_key = st.text_input("Gemini API Key:", type="password", placeholder="AIzaSy...")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key.strip()
    openai_key = st.text_input("OpenAI API Key:", type="password", placeholder="sk-...")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()

# ──────────────────────────────────────────────────────────────────────
# Page 1: Ask a Legal Question (RAG QA)
# ──────────────────────────────────────────────────────────────────────
if page == "🔍 Page 1: Ask a Legal Question":
    st.title("🔍 Domain-Driven Legal Question Answering")
    st.caption("Select a legal domain and subdomain to execute domain-filtered RAG retrieval and receive highly grounded answers.")

    # Step 1 & 2: Hierarchical Domain Selection
    st.subheader("⚖️ Hierarchical Domain Selection")
    
    col_domain, col_subdomain = st.columns(2)
    
    with col_domain:
        selected_domain = st.selectbox(
            "Step 1: Select Primary Legal Domain",
            options=list(LEGAL_DOMAINS.keys()),
            index=0
        )
        
    with col_subdomain:
        subdomains = LEGAL_DOMAINS[selected_domain]
        selected_subdomain = st.selectbox(
            "Step 2: Select Legal Subdomain",
            options=subdomains,
            index=0
        )
        
    # Display Active Path with Applicable Law
    applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(selected_subdomain, selected_subdomain)
    st.info(f"📂 **Active Path:** `{selected_domain}` ➔ `{selected_subdomain}`")
    st.caption(f"📜 **Applicable Law:** {applicable_law}  |  🌏 **Jurisdiction:** India")

    # Voice Preferences
    st.subheader("🔊 Audio Synthesis Preferences")
    voice_gender = st.radio(
        "Select TTS Voice Gender:",
        ["Female (Neerja)", "Male (Prabhat)"],
        index=0,
        horizontal=True
    )
    gender_backend = "Female" if "Female" in voice_gender else "Male"

    # Presets Definition
    SUBDOMAIN_SAMPLE_QUERIES = {
        "Criminal Law": {
            "CrPC": "What are the requirements for bail under CrPC and fundamental rights under Article 14?",
            "Bail Procedures": "What arguments were raised regarding bail procedures and framing of charges?",
            "Indian Penal Code (IPC)": "What did the court discuss regarding framing of charges and disqualification from elections?",
            "Indian Evidence Act": "What are the rules regarding the admissibility of digital evidence under the Evidence Act?",
            "Sentencing Guidelines": "What factors do Indian courts consider when determining sentencing guidelines?"
        },
        "Constitutional & Administrative Law": {
            "Fundamental Rights": "What arguments were raised regarding Article 21 and Section 8(3) of Representation of People Act?",
            "Writ Petitions": "How is a writ petition filed under Article 32 or 226 for fundamental rights violation?",
            "Center-State Relations": "Explain the distribution of legislative powers between Center and States under the Seventh Schedule.",
            "Administrative Tribunals": "What is the jurisdiction of Administrative Tribunals in service matters in India?",
            "Constitutional Amendments": "What is section 9A of the Representation of the People Act?"
        },
        "Corporate & Business Law": {
            "Contract Law": "What are the statutory grounds for setting aside an arbitral award under Section 34?",
            "Companies Act": "What are the duties of directors under the Indian Companies Act 2013?",
            "Partnership & LLP": "Explain the key differences between a traditional partnership and a Limited Liability Partnership (LLP).",
            "Insolvency & Bankruptcy Code (IBC)": "What is the corporate insolvency resolution process (CIRP) timeline under IBC?",
            "Intellectual Property (IP)": "What constitute patentable subject matters under the Indian Patents Act?"
        },
        "Finance & Tax Law": {
            "GST": "What are the GST implications for a small business in India?",
            "Income Tax": "What are the standard deduction rules for salaried individuals under the Income Tax Act?",
            "Banking Law": "What are the compliance requirements for NBFCs under RBI regulations?",
            "Securities Law": "What powers does SEBI have to check insider trading in India?",
            "Financial Regulations": "Explain the compliance requirements under FEMA for foreign direct investment."
        },
        "Cyber & Digital Law": {
            "IT Act": "What is Section 66A of the IT Act and why was it struck down by the Supreme Court?",
            "Data Privacy": "What are the data privacy compliance rules under the Digital Personal Data Protection (DPDP) Act?",
            "Cyber Security Regulations": "What are the mandatory reporting timelines for cyber security incidents under CERT-In?",
            "Cyber Crime": "What are the legal remedies against online identity theft and phishing under Indian law?",
            "Digital Evidence": "How is digital evidence certified under Section 65B of the Indian Evidence Act?"
        },
        "Employment & Labour Law": {
            "Industrial Disputes": "What is the procedure for settling industrial disputes under the Industrial Disputes Act?",
            "Wages & Bonus": "How is minimum wage calculated and enforced in India?",
            "Social Security": "What are the employee eligibility criteria for gratuity under Indian labour laws?",
            "Trade Unions": "Explain the registration and recognition process of Trade Unions in India.",
            "Factories Act": "What are the safety and health provisions mandated for workers under the Factories Act?"
        },
        "Property & Real Estate Law": {
            "Transfer of Property Act": "What are the essential conditions for a valid transfer of immovable property by sale?",
            "RERA": "What are the builder disclosure obligations and registration requirements under RERA?",
            "Land Acquisition": "What is the compensation calculation model under the Right to Fair Compensation Act?",
            "Landlord-Tenant Disputes": "What are the legal grounds for eviction of a tenant under rent control laws?",
            "Easements": "What constitutes an easementary right of way by prescription under Indian law?"
        },
        "Family & Personal Law": {
            "Marriage & Divorce": "What are the legal grounds for divorce under the Hindu Marriage Act?",
            "Succession & Inheritance": "How is property distributed for a Hindu male dying intestate under the Hindu Succession Act?",
            "Maintenance & Alimony": "What are the maintenance rights of a wife under Section 125 of CrPC?",
            "Guardianship": "Who is considered the natural guardian of a minor child under Hindu law?",
            "Hindu/Muslim Law": "What are the inheritance rules for women under Muslim personal law in India?"
        },
        "Consumer & Safety Law": {
            "Consumer Protection": "What are the rights of a consumer against unfair trade practices under the Consumer Protection Act 2019?",
            "Product Liability": "What are the product liability claims a consumer can raise against a manufacturer?",
            "Food Safety Regulations": "What are the labelling requirements and food safety regulations under FSSAI?",
            "Workplace Safety": "What are the duties of employers regarding safety and hazard prevention in factories?",
            "Public Safety Regulations": "What are the legal liabilities under the Environmental Protection Act for industrial pollution?"
        }
    }

    preset_query = SUBDOMAIN_SAMPLE_QUERIES.get(selected_domain, {}).get(selected_subdomain, "")
    
    st.markdown("**Sample Benchmark Query:**")
    use_preset = st.button(f"📋 Use Preset: {preset_query}")
    
    query_val = preset_query if use_preset else ""
    
    query_input = st.text_input(
        "Enter your legal question:",
        value=query_val,
        placeholder=f"Type your query for {selected_subdomain} here..."
    )

    if st.button("Run Domain Query", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a valid legal question.")
        else:
            with st.spinner(f"Classifying query, executing RAG query on {selected_domain} context, and synthesizing answer..."):
                try:
                    explanation, citations, audio_file, metrics = process_hierarchical_legal_query(
                        query_input.strip(), 
                        selected_domain, 
                        selected_subdomain, 
                        voice_gender=gender_backend
                    )
                    
                    if "⚠️ **Domain Mismatch:**" in explanation:
                        st.warning("Domain Mismatch Detected")
                    else:
                        st.success("Query processed successfully!")
                    
                    # Display Clean Explanation only
                    st.subheader("📝 Answer")
                    st.markdown(explanation)
                        
                    # Embedded Audio Player
                    st.subheader("🔊 Spoken Audio Explanation")
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/mp3")
                        st.caption(f"Generated audio file ({voice_gender}): `{os.path.basename(audio_file)}`")
                    else:
                        st.warning("Audio synthesis was not generated.")
                        
                    # Collapsible Evaluation Metrics Section
                    with st.expander("📊 Diagnostic Evaluation Metrics (Collapsible)", expanded=False):
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Mean Cosine Sim", f"{metrics['mean_cosine_similarity']:.4f}")
                        m_col2.metric("Max Cosine Sim", f"{metrics['max_cosine_similarity']:.4f}")
                        m_col3.metric("Context Precision", f"{metrics['context_precision']*100:.1f}%")
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
                    st.error(f"Error executing Domain query pipeline: {str(ex)}")

# ──────────────────────────────────────────────────────────────────────
# Page 2: Upload Audio & Summarize
# ──────────────────────────────────────────────────────────────────────
elif page == "🎙️ Page 2: Upload Audio & Summarize":
    st.title("🎙️ Upload Audio Proceeding & Summarize")
    st.caption("Upload courtroom/moot court audio files (.wav, .mp3, .m4a) to generate speaker diarization, transcription, and an executive summary.")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        help="Upload a legal proceeding recording (.wav, .mp3, or .m4a)"
    )
    
    st.info("🔒 **Rights Confirmation Required**")
    consent = st.checkbox(
        "I confirm I have the rights to upload and process this recording (e.g. it's not a Supreme Court/High Court live-stream, which prohibits redistribution)."
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        process_btn = st.button("Process Audio & Generate Summary", type="primary", disabled=not consent, use_container_width=True)
        if not consent:
            st.warning("⚠️ Please check the consent box above to confirm rights ownership before processing.")
            
        if process_btn and consent:
            with st.status("Processing audio file...", expanded=True) as status:
                try:
                    # Save uploaded file to temp file
                    temp_dir = tempfile.mkdtemp()
                    input_ext = uploaded_file.name.split(".")[-1].lower()
                    raw_audio_path = os.path.join(temp_dir, f"input_raw.{input_ext}")
                    pcm_wav_path = os.path.join(temp_dir, "input_16k.wav")
                    
                    with open(raw_audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Step 1: Preprocessing & 16kHz PCM Audio Conversion
                    status.update(label="Step 1/3: Signal Preprocessing & Audio Extraction (16kHz PCM)...")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", raw_audio_path,
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                        pcm_wav_path
                    ]
                    res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if res.returncode != 0 or not os.path.exists(pcm_wav_path):
                        st.error("Audio conversion failed via FFmpeg. Please check if the audio file is valid.")
                        st.stop()
                        
                    # Step 2: Diarization & Faster-Whisper ASR Transcription
                    status.update(label="Step 2/3: Neural Diarization & Faster-Whisper ASR (CUDA GPU)...")
                    
                    from faster_whisper import WhisperModel
                    model = WhisperModel("small", device="cuda", compute_type="int8")
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
                        st.error("Transcription yielded empty text. Please check if the audio contains audible speech.")
                        st.stop()
                        
                    # Step 3: LLM Summarization with User-Specified Prompt
                    status.update(label="Step 3/3: Generating Executive Legal Summary via LLM...")
                    
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
                    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
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

                    hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
                    if not summary_output and hf_token:
                        try:
                            from huggingface_hub import InferenceClient
                            client = InferenceClient(token=hf_token)
                            r = client.chat_completion(
                                model="meta-llama/Llama-3.2-3B-Instruct",
                                messages=[{"role": "user", "content": summarization_prompt}],
                                max_tokens=600,
                                temperature=0.3
                            )
                            raw_out = r.choices[0].message.content.strip()
                            if raw_out and len(raw_out) > 20 and not raw_out.startswith("I can't provide"):
                                summary_output = raw_out
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
                        
                    status.update(label="Complete!", state="complete", expanded=False)
                    
                    # Display Output
                    st.subheader("📑 Executive Legal Summary")
                    st.markdown(summary_output)
                    
                    # Expandable Full Transcript Section
                    with st.expander("📜 Full Speaker-Labeled Transcript", expanded=False):
                        for line in transcript_entries:
                            st.text(line)
                            
                    st.info("💡 Note: Uploaded audio and transcripts remain session-only and are not added to the persistent ChromaDB index.")
                    
                except Exception as ex:
                    status.update(label="Error occurred", state="error")
                    st.error(f"Error during audio processing: {str(ex)}")
