import os
import sys
import json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

try:
    import spaces
    @spaces.GPU(duration=30)
    def gpu_warmup():
        return True
    gpu_warmup()
except Exception:
    pass

from legal_rag_qa import (
    process_hierarchical_legal_query,
    explain_legal_answer_simple,
    LEGAL_DOMAINS,
    SUBDOMAIN_APPLICABLE_LAW
)

def classify_legal_question(query):
    """Infer a useful legal path from the user's wording without requiring legal taxonomy knowledge."""
    text = (query or "").lower()
    rules = [
        ("Constitutional & Administrative Law", "Election & Representation Law", [
            "section 9a", "section 8", "section 8a", "representation of the people",
            "representation of people", "rpa", "election", "elections", "electoral"
        ]),
        ("Constitutional & Administrative Law", "Fundamental Rights", [
            "article 14", "article 19", "article 21", "article 32", "article 226",
            "fundamental right", "fundamental rights", "constitution"
        ]),
        ("Criminal Law", "CrPC", ["crpc", "criminal procedure", "fir", "arrest", "framing of charge"]),
        ("Criminal Law", "Bail Procedures", ["bail", "non-bailable", "non bailable"]),
        ("Criminal Law", "Indian Evidence Act", ["evidence act", "electronic evidence", "section 65b"]),
        ("Corporate & Business Law", "Contract Law", ["arbitration", "arbitral award", "section 34"]),
        ("Cyber & Digital Law", "IT Act", ["it act", "information technology", "section 66a", "section 79"]),
        ("Constitutional & Administrative Law", "Election & Representation Law", ["disqualification", "government contract"]),
    ]
    for domain, subdomain, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return domain, subdomain
    return "Constitutional & Administrative Law", "Fundamental Rights"


def format_detected_area_html(domain, subdomain):
    applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(subdomain, subdomain)
    return f"""
<div class="detected-badge-card">
    <div class="badge-header-row">
        <div class="badge-status-wrap">
            <span class="live-pulse-dot"></span>
            <span class="badge-pill-title">Domain Router</span>
        </div>
        <span class="badge-domain-pill">{domain}</span>
    </div>
    <div class="badge-sub-row">
        <span class="badge-arrow">&gt;</span>
        <span class="badge-subdomain">{subdomain}</span>
    </div>
    <div class="badge-law-row">
        <span class="badge-law-text"><strong>Governing Statute:</strong> {applicable_law}</span>
    </div>
</div>
"""


def format_applicable_law_card(domain, subdomain):
    applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(subdomain, subdomain)
    return f"""
<div class="law-context-card">
    <div class="law-context-item">
        <span class="context-label">Enactment</span>
        <span class="context-val">{applicable_law}</span>
    </div>
    <div class="law-context-sep"></div>
    <div class="law-context-item">
        <span class="context-label">Domain</span>
        <span class="context-val">{domain} &gt; {subdomain}</span>
    </div>
    <div class="law-context-sep"></div>
    <div class="law-context-item">
        <span class="context-label">Jurisdiction</span>
        <span class="context-val">Republic of India</span>
    </div>
</div>
"""


def update_detected_area(query):
    domain, subdomain = classify_legal_question(query)
    card_html = format_detected_area_html(domain, subdomain)
    return card_html, domain, subdomain


def handle_gradio_query(query, detected_domain, detected_subdomain, manual_mode, manual_domain, manual_subdomain, voice_gender):
    if not query or not query.strip():
        empty_law = format_applicable_law_card("Constitutional & Administrative Law", "Fundamental Rights")
        return "Please enter a legal question to initiate statutory analysis.", empty_law, "", None, gr.update(interactive=False), gr.update(value="", visible=False)
    
    try:
        domain = manual_domain if manual_mode else detected_domain
        subdomain = manual_subdomain if manual_mode else detected_subdomain
        
        clean_gender = "Female"
        if voice_gender and "Male" in str(voice_gender):
            clean_gender = "Male"

        # Run the 21-rule RAG retrieval & synthesis pipeline
        ans, cites, aud_path, metrics = process_hierarchical_legal_query(
            query_text=query.strip(),
            domain=domain,
            subdomain=subdomain,
            voice_gender=clean_gender
        )
        is_valid = bool(ans and not ans.startswith("Please enter") and not ans.startswith("Error") and not ans.startswith("\u26a0"))
        
        law_card_html = format_applicable_law_card(domain, subdomain)
        
        return ans, law_card_html, cites, aud_path, gr.update(interactive=is_valid), gr.update(value="", visible=False)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        error_msg = f"**An error occurred during query processing:** {str(e)}\n\nPlease try rephrasing your legal question or selecting the domain again."
        empty_law = format_applicable_law_card(detected_domain, detected_subdomain)
        return error_msg, empty_law, "Execution Error", None, gr.update(interactive=False), gr.update(value="", visible=False)


def handle_explain_simple(verified_answer, citations):
    if not verified_answer or not verified_answer.strip() or verified_answer.startswith("Please enter") or verified_answer.startswith("Error") or verified_answer.startswith("\u26a0"):
        return gr.update(value="<div class='advisory-empty'>Generate a verified legal answer first to view the client-friendly explanation.</div>", visible=True)
    explanation = explain_legal_answer_simple(verified_answer, citations)
    formatted = f"""### Client Advisory: Plain-Language Explanation\n\n{explanation}"""
    return gr.update(value=formatted, visible=True)


# Default subdomains for initial load
default_domain = "Constitutional & Administrative Law"
default_subs = LEGAL_DOMAINS.get(default_domain, ["Fundamental Rights"])
all_subdomains = sorted(list({s for subs in LEGAL_DOMAINS.values() for s in subs}))
initial_detected_html = format_detected_area_html(default_domain, default_subs[0] if default_subs else "Fundamental Rights")
initial_law_card_html = format_applicable_law_card(default_domain, default_subs[0] if default_subs else "Fundamental Rights")

# Sample questions matching exact statutory framework
SAMPLE_QUESTIONS = [
    {
        "label": "Sec 9A RPA 1951",
        "sublabel": "Contract Disqualification",
        "query": "What does Section 9A of the Representation of the People Act, 1951 state regarding disqualification for government contracts?"
    },
    {
        "label": "Art 21 Constitution",
        "sublabel": "Life & Personal Liberty",
        "query": "How is the right to life and personal liberty protected under Article 21 of the Constitution of India in Supreme Court jurisprudence?"
    },
    {
        "label": "Sec 65B Evidence Act",
        "sublabel": "Electronic Record Certificate",
        "query": "What are the mandatory certificate requirements for electronic records admissibility under Section 65B of the Indian Evidence Act, 1872?"
    },
    {
        "label": "Sec 437 CrPC",
        "sublabel": "Bail in Non-Bailable Cases",
        "query": "What statutory conditions and judicial principles govern the grant of bail under Section 437 of the Code of Criminal Procedure, 1973 for non-bailable offences?"
    },
    {
        "label": "Sec 34 Arbitration Act",
        "sublabel": "Setting Aside Arbitral Award",
        "query": "On what statutory grounds can an arbitral award be set aside under Section 34 of the Arbitration and Conciliation Act, 1996?"
    }
]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #090d16;
    --bg-surface: #0e1628;
    --bg-elevated: #131e36;
    --border-subtle: rgba(148, 163, 184, 0.12);
    --border-focus: rgba(59, 130, 246, 0.45);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #2563eb;
    --accent-blue-hover: #1d4ed8;
    --accent-teal: #0d9488;
    --accent-amber: #d97706;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 16px 24px 44px !important;
}

/* ==========================================================================
   Header and System Specification Bar
   ========================================================================== */
.studio-hero {
    background: linear-gradient(135deg, rgba(14, 22, 40, 0.95) 0%, rgba(10, 16, 30, 0.98) 100%) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 22px 26px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

.hero-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

.brand-identity {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon-svg {
    width: 38px;
    height: 38px;
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #60a5fa;
}

.brand-title-wrap {
    display: flex;
    flex-direction: column;
}

.brand-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: clamp(1.4rem, 2.2vw, 1.85rem) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    margin: 0 !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
}

.brand-subtitle {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    margin: 2px 0 0 0 !important;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(13, 148, 136, 0.12);
    border: 1px solid rgba(13, 148, 136, 0.35);
    color: #2dd4bf;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #14b8a6;
}

/* System Capabilities Grid (clearly labeled as configured specs) */
.capabilities-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 10px;
    padding-top: 14px;
    border-top: 1px solid var(--border-subtle);
}

.capability-item {
    background: rgba(19, 30, 54, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 8px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
}

.capability-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    font-weight: 600;
    color: #93c5fd;
}

.capability-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 1px;
}

/* ==========================================================================
   Panel Containers
   ========================================================================== */
.query-panel, .answer-panel {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 22px !important;
    box-shadow: 0 10px 24px -8px rgba(0, 0, 0, 0.4) !important;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

.panel-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.01em;
}

.verified-tag {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #34d399;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 2px 8px;
    border-radius: 6px;
}

/* ==========================================================================
   Query Workspace Inputs
   ========================================================================== */
.gradio-container textarea, .gradio-container input[type="text"] {
    background-color: var(--bg-base) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.94rem !important;
    line-height: 1.5 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
}

.gradio-container label span {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
}

/* Primary Execution Button */
.primary-action-btn {
    min-height: 48px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
    cursor: pointer !important;
    margin-top: 12px !important;
}

.primary-action-btn:hover {
    background: var(--accent-blue-hover) !important;
    transform: translateY(-1px) !important;
}

.primary-action-btn:active {
    transform: translateY(0) !important;
}

/* Suggested Query Chips */
.chips-section {
    margin: 12px 0 14px !important;
}

.chips-header {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.chip-btn {
    background: rgba(19, 30, 54, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.chip-btn:hover {
    background: rgba(37, 99, 235, 0.15) !important;
    border-color: rgba(59, 130, 246, 0.4) !important;
    color: #ffffff !important;
}

/* Dynamic Domain Router Card */
.detected-badge-card {
    background: rgba(14, 22, 40, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 12px 14px;
    margin: 12px 0;
}

.badge-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

.badge-status-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
}

.live-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #3b82f6;
}

.badge-pill-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    color: var(--text-muted);
}

.badge-domain-pill {
    background: rgba(37, 99, 235, 0.14);
    border: 1px solid rgba(37, 99, 235, 0.3);
    color: #93c5fd;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
}

.badge-sub-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    color: var(--text-primary);
    font-size: 0.88rem;
}

.badge-arrow {
    color: #60a5fa;
    font-weight: 700;
}

.badge-subdomain {
    font-weight: 600;
}

.badge-law-row {
    font-size: 0.8rem;
    color: var(--text-secondary);
    background: rgba(9, 13, 22, 0.5);
    padding: 5px 8px;
    border-radius: 6px;
    border-left: 2px solid var(--accent-blue);
}

/* Applicable Law Banner Card in Right Column */
.law-context-card {
    background: rgba(14, 22, 40, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}

.law-context-item {
    display: flex;
    flex-direction: column;
}

.context-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
}

.context-val {
    font-size: 0.82rem;
    font-weight: 600;
    color: #e2e8f0;
}

.law-context-sep {
    width: 1px;
    height: 18px;
    background: var(--border-subtle);
}

/* ==========================================================================
   Answer and Evidence Workspace
   ========================================================================== */
.answer-panel textarea {
    font-size: 0.94rem !important;
    line-height: 1.6 !important;
}

.explain-action-btn {
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    min-height: 40px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    background: rgba(217, 119, 6, 0.1) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(217, 119, 6, 0.3) !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.explain-action-btn:enabled:hover {
    background: rgba(217, 119, 6, 0.18) !important;
    border-color: #d97706 !important;
}

.simple-explain-card {
    background: rgba(217, 119, 6, 0.05) !important;
    border: 1px solid rgba(217, 119, 6, 0.2) !important;
    border-left: 3px solid #d97706 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin-bottom: 14px !important;
    font-size: 0.92rem !important;
    line-height: 1.55 !important;
}

.citations-card {
    background: rgba(9, 13, 22, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin-top: 10px !important;
    margin-bottom: 12px !important;
    font-size: 0.88rem !important;
}

.audio-card {
    background: rgba(9, 13, 22, 0.5) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 12px 12px 28px !important;
    }
    .capabilities-bar {
        grid-template-columns: 1fr 1fr;
    }
}
"""

# Create Gradio Interface
with gr.Blocks(title="Indian Legal Speech RAG Studio", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    # -- HEADER & CONFIGURED ARCHITECTURE BAR --
    with gr.Group(elem_classes="studio-hero"):
        gr.HTML("""
        <div class="hero-container">
            <div class="hero-header-row">
                <div class="brand-identity">
                    <div class="brand-icon-svg">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
                            <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
                            <path d="M7 21h10"/>
                            <path d="M12 3v18"/>
                            <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>
                        </svg>
                    </div>
                    <div class="brand-title-wrap">
                        <h1 class="brand-title">Indian Legal Speech RAG Studio</h1>
                        <p class="brand-subtitle">Grounded statutory synthesis and multi-speaker courtroom retrieval with neural voice narration</p>
                    </div>
                </div>
                <div class="status-pill">
                    <span class="status-dot"></span> Pipeline Ready
                </div>
            </div>
            
            <div class="capabilities-bar">
                <div class="capability-item">
                    <span class="capability-val">4,260+ Chunks</span>
                    <span class="capability-lbl">Statutory Provisions & Courtroom Turns</span>
                </div>
                <div class="capability-item">
                    <span class="capability-val">BGE-Large 1024-D</span>
                    <span class="capability-lbl">Dense Semantic Embedding Model</span>
                </div>
                <div class="capability-item">
                    <span class="capability-val">21-Rule Engine</span>
                    <span class="capability-lbl">Factual Grounding & Citation Verifier</span>
                </div>
                <div class="capability-item">
                    <span class="capability-val">Neural TTS (en-IN)</span>
                    <span class="capability-lbl">Neerja & Prabhat Voice Synthesis</span>
                </div>
            </div>
        </div>
        """)
    
    with gr.Row():
        # -- LEFT COLUMN: QUERY WORKSPACE --
        with gr.Column(scale=5, elem_classes="query-panel"):
            gr.HTML("""
            <div class="panel-header">
                <span class="panel-header-title">Legal Query Workspace</span>
            </div>
            """)

            query_box = gr.Textbox(
                lines=4,
                label="Enter Legal Question / Statutory Issue",
                placeholder="State your legal inquiry (e.g., What does Section 9A of the Representation of the People Act, 1951 state regarding disqualification for government contracts?)"
            )

            # Compact Benchmark Query Chips
            with gr.Group(elem_classes="chips-section"):
                gr.HTML('<div class="chips-header">Suggested Landmark Inquiries:</div>')
                with gr.Row():
                    chip_btns = []
                    for item in SAMPLE_QUESTIONS:
                        btn = gr.Button(item["label"], size="sm", elem_classes="chip-btn")
                        chip_btns.append((btn, item["query"]))

            # Dynamic AI Domain Router Card
            detected_area = gr.HTML(initial_detected_html)
            detected_domain = gr.Textbox(value=default_domain, visible=False)
            detected_subdomain = gr.Textbox(value=default_subs[0] if default_subs else "Fundamental Rights", visible=False)

            # Wire query typing to update classifier
            query_box.input(
                update_detected_area,
                inputs=[query_box],
                outputs=[detected_area, detected_domain, detected_subdomain]
            )

            # Wire chip clicks to fill query and update classifier
            for btn, sample_q in chip_btns:
                btn.click(
                    fn=lambda q=sample_q: (q, *update_detected_area(q)),
                    inputs=[],
                    outputs=[query_box, detected_area, detected_domain, detected_subdomain]
                )

            # Collapsible Advanced Taxonomy Overrides
            with gr.Accordion("Advanced Options: Manual Taxonomy Override", open=False):
                manual_mode = gr.Checkbox(label="Enable manual domain override", value=False)
                domain_dropdown = gr.Dropdown(
                    choices=list(LEGAL_DOMAINS.keys()),
                    value=default_domain,
                    label="Primary Legal Domain"
                )
                subdomain_dropdown = gr.Dropdown(
                    choices=LEGAL_DOMAINS[default_domain],
                    value=default_subs[0] if default_subs else None,
                    label="Legal Subdomain"
                )

                def on_domain_change(d):
                    subs = LEGAL_DOMAINS.get(d, [])
                    return gr.update(choices=subs, value=subs[0] if subs else None)

                domain_dropdown.change(on_domain_change, inputs=[domain_dropdown], outputs=[subdomain_dropdown])

            voice_choice = gr.Radio(
                choices=[("Female (Neerja)", "Female"), ("Male (Prabhat)", "Male")],
                value="Female",
                label="Neural Voice Synthesis Voiceover"
            )

            submit_btn = gr.Button("Analyze Legal Query", variant="primary", elem_classes="primary-action-btn")
            
        # -- RIGHT COLUMN: LEGAL ANSWER & EVIDENCE WORKSPACE --
        with gr.Column(scale=6, elem_classes="answer-panel"):
            gr.HTML("""
            <div class="panel-header">
                <span class="panel-header-title">Legal Analysis & Evidence Workspace</span>
                <span class="verified-tag">Grounded Synthesis</span>
            </div>
            """)

            # Section 2: Applicable Law Banner
            applicable_law_card = gr.HTML(initial_law_card_html)

            # Section 1: Grounded Legal Answer
            answer_box = gr.Textbox(
                lines=8,
                label="Legal Analysis",
                placeholder="The grounded statutory analysis will appear here once the query is processed..."
            )

            # Section 5: Client Advisory (Plain-Language Explanation)
            explain_btn = gr.Button("Explain in Simple Words (Client Advisory)", interactive=False, elem_classes="explain-action-btn")
            simple_explain_box = gr.Markdown(visible=False, elem_classes="simple-explain-card")

            # Section 3 & 4: Supporting Evidence, Citations & Provenance
            citations_box = gr.Markdown(label="Supporting Evidence, Citations & Provenance", elem_classes="citations-card")

            # Section 6: Spoken Neural Voice Output
            audio_box = gr.Audio(label="Spoken Neural Voice Explanation (Indian English)", type="filepath", elem_classes="audio-card")

    # Wire Submit Action
    submit_btn.click(
        handle_gradio_query,
        inputs=[query_box, detected_domain, detected_subdomain, manual_mode, domain_dropdown, subdomain_dropdown, voice_choice],
        outputs=[answer_box, applicable_law_card, citations_box, audio_box, explain_btn, simple_explain_box]
    )

    # Wire Explain Action
    explain_btn.click(
        handle_explain_simple,
        inputs=[answer_box, citations_box],
        outputs=[simple_explain_box]
    )

# Launch the Gradio app directly with show_api=False to prevent JSON schema parse errors
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)


