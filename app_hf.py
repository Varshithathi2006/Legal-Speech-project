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
            <span class="badge-pill-title">AI Domain Router</span>
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


def update_detected_area(query):
    domain, subdomain = classify_legal_question(query)
    card_html = format_detected_area_html(domain, subdomain)
    return card_html, domain, subdomain


def handle_gradio_query(query, detected_domain, detected_subdomain, manual_mode, manual_domain, manual_subdomain, voice_gender):
    if not query or not query.strip():
        return "Please enter a legal question to initiate statutory search.", "", None, gr.update(interactive=False), gr.update(value="", visible=False)
    
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
        return ans, cites, aud_path, gr.update(interactive=is_valid), gr.update(value="", visible=False)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        error_msg = f"**An error occurred during query processing:** {str(e)}\n\nPlease try rephrasing your legal question or selecting the domain again."
        return error_msg, "N/A - Execution Error", None, gr.update(interactive=False), gr.update(value="", visible=False)


def handle_explain_simple(verified_answer, citations):
    if not verified_answer or not verified_answer.strip() or verified_answer.startswith("Please enter") or verified_answer.startswith("Error") or verified_answer.startswith("\u26a0"):
        return gr.update(value="Please generate a valid verified legal answer first.", visible=True)
    explanation = explain_legal_answer_simple(verified_answer, citations)
    formatted = f"""### Client Advisory Summary\n\n{explanation}"""
    return gr.update(value=formatted, visible=True)


# Default subdomains for initial load
default_domain = "Constitutional & Administrative Law"
default_subs = LEGAL_DOMAINS.get(default_domain, ["Fundamental Rights"])
all_subdomains = sorted(list({s for subs in LEGAL_DOMAINS.values() for s in subs}))
initial_detected_html = format_detected_area_html(default_domain, default_subs[0] if default_subs else "Fundamental Rights")

# Sample questions for quick one-click testing
SAMPLE_QUESTIONS = [
    {
        "label": "Sec 9A RPA (Govt Contracts)",
        "query": "What does Section 9A of the Representation of the People Act state regarding disqualification for government contracts?"
    },
    {
        "label": "Art 21 (Right to Privacy)",
        "query": "How is the Right to Privacy protected under Article 21 of the Constitution of India in landmark Supreme Court jurisprudence?"
    },
    {
        "label": "Sec 65B Evidence Act (Electronic)",
        "query": "What are the mandatory certificate requirements for electronic records admissibility under Section 65B of the Indian Evidence Act?"
    },
    {
        "label": "Sec 437 CrPC (Bail Criteria)",
        "query": "What statutory conditions and judicial principles govern the grant of bail under Section 437 of the Code of Criminal Procedure for non-bailable offences?"
    },
    {
        "label": "Sec 34 Arbitration (Setting Aside)",
        "query": "On what legal grounds can an arbitral award be set aside under Section 34 of the Arbitration and Conciliation Act?"
    }
]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary-color: #3b82f6;
    --primary-hover: #2563eb;
    --surface-dark: #0f172a;
    --surface-card: #131d33;
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-highlight: rgba(59, 130, 246, 0.4);
    --accent-gold: #f59e0b;
    --accent-emerald: #10b981;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #080c16 !important;
    color: #f1f5f9 !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 24px 28px 48px !important;
}

/* ==========================================================================
   Hero Header Bar
   ========================================================================== */
.studio-hero {
    background: linear-gradient(135deg, rgba(19, 29, 51, 0.85) 0%, rgba(13, 20, 36, 0.95) 100%) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 18px !important;
    padding: 24px 30px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
}

.hero-container {
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 18px;
}

.hero-badge-icon {
    font-size: 2.5rem;
    line-height: 1;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 14px;
    padding: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.3);
}

.hero-title-group {
    flex: 1;
}

.hero-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: clamp(1.6rem, 2.6vw, 2.2rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    margin: 0 !important;
    background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.status-live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.pulse-indicator-emerald {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 0 rgba(16, 185, 129, 0.6);
    animation: pulse-emerald 2s infinite;
}

@keyframes pulse-emerald {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.hero-tagline {
    color: #94a3b8 !important;
    font-size: 0.98rem !important;
    margin: 0 !important;
    line-height: 1.45 !important;
}

.hero-stats-bar {
    display: flex;
    align-items: center;
    gap: 20px;
    padding-top: 14px;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    flex-wrap: wrap;
}

.hero-stat-box {
    display: flex;
    flex-direction: column;
}

.stat-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #60a5fa;
}

.stat-label {
    font-size: 0.74rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

.hero-stat-sep {
    width: 1px;
    height: 24px;
    background: rgba(148, 163, 184, 0.15);
}

/* ==========================================================================
   Panels (Left: Query Console, Right: Verified Answer)
   ========================================================================== */
.query-panel, .answer-panel {
    background: rgba(17, 26, 46, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.25s ease !important;
}

.query-panel:hover, .answer-panel:hover {
    border-color: rgba(59, 130, 246, 0.25) !important;
}

.panel-header-custom {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.panel-header-custom .panel-icon {
    font-size: 1.25rem;
}

/* Textboxes and Textareas */
.gradio-container textarea, .gradio-container input[type="text"] {
    background-color: #0b1120 !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.96rem !important;
    transition: all 0.2s ease !important;
}

.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

/* Labels */
.gradio-container label span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: #cbd5e1 !important;
    text-transform: uppercase !important;
}

/* Quick prompt chips */
.sample-chips-wrap {
    margin: 10px 0 16px !important;
}

.chips-label {
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
    color: #64748b !important;
    margin-bottom: 8px !important;
}

.sample-chip-btn {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    border-radius: 18px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}

.sample-chip-btn:hover {
    background: rgba(59, 130, 246, 0.22) !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px -2px rgba(59, 130, 246, 0.3) !important;
}

/* AI Detected Domain Badge Card */
.detected-badge-card {
    background: linear-gradient(135deg, rgba(20, 32, 56, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 12px 0 16px;
    box-shadow: 0 6px 18px -4px rgba(0, 0, 0, 0.35);
}

.badge-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.badge-status-wrap {
    display: flex;
    align-items: center;
    gap: 7px;
}

.live-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    box-shadow: 0 0 0 rgba(59, 130, 246, 0.6);
    animation: pulse-blue 2s infinite;
}

@keyframes pulse-blue {
    0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
    70% { box-shadow: 0 0 0 6px rgba(59, 130, 246, 0); }
    100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
}

.badge-pill-title {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    color: #94a3b8;
}

.badge-domain-pill {
    background: rgba(59, 130, 246, 0.18);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
}

.badge-sub-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    color: #e2e8f0;
    font-size: 0.92rem;
}

.badge-arrow {
    color: #60a5fa;
    font-size: 0.88rem;
}

.badge-subdomain {
    font-weight: 600;
}

.badge-law-row {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    font-size: 0.84rem;
    color: #94a3b8;
    background: rgba(11, 17, 32, 0.5);
    padding: 6px 10px;
    border-radius: 8px;
    border-left: 2px solid #3b82f6;
}

.badge-law-text strong {
    color: #cbd5e1;
}

/* Primary Execution Button */
.primary-search-btn {
    margin-top: 14px !important;
    min-height: 52px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px -2px rgba(37, 99, 235, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}

.primary-search-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px -2px rgba(37, 99, 235, 0.65) !important;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
}

.primary-search-btn:active {
    transform: translateY(0) !important;
}

/* Explain Button */
.explain-action-btn {
    margin-top: 14px !important;
    margin-bottom: 12px !important;
    min-height: 44px !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.08) 100%) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.35) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.explain-action-btn:enabled:hover {
    background: rgba(245, 158, 11, 0.22) !important;
    border-color: #f59e0b !important;
    color: #fef3c7 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px -2px rgba(245, 158, 11, 0.3) !important;
}

/* Simple Explanation Container */
.simple-explain-card {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.06) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin-bottom: 16px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* Citations Container */
.citations-card {
    background: rgba(11, 17, 32, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin-top: 14px !important;
    margin-bottom: 16px !important;
}

/* Audio Player */
.audio-card {
    background: rgba(11, 17, 32, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
}

/* Radio Choice Styling */
.gradio-container .gr-radio {
    gap: 8px !important;
}

.gradio-container .gr-radio label {
    background: rgba(11, 17, 32, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease !important;
}

.gradio-container .gr-radio label:hover {
    border-color: rgba(59, 130, 246, 0.35) !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 16px 14px 32px !important;
    }
    .hero-stats-bar {
        display: none !important;
    }
    .hero-brand {
        flex-direction: column !important;
        align-items: flex-start !important;
    }
}
"""

# Create Gradio Interface
with gr.Blocks(title="Indian Legal Speech RAG Studio & API", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    # Executive Hero Banner
    with gr.Group(elem_classes="studio-hero"):
        gr.HTML("""
        <div class="hero-container">
            <div class="hero-brand">
                <div class="hero-title-group">
                    <div class="hero-title-row">
                        <h1 class="hero-title">Indian Legal Speech RAG Studio</h1>
                        <span class="status-live-pill"><span class="pulse-indicator-emerald"></span> Production Pipeline</span>
                    </div>
                    <p class="hero-tagline">
                        Grounded statutory synthesis from Indian penal and constitutional codes with multi-speaker courtroom retrieval and neural voice narration
                    </p>
                </div>
            </div>
            <div class="hero-stats-bar">
                <div class="hero-stat-box">
                    <span class="stat-number">4,260+</span>
                    <span class="stat-label">Statutory & Speech Chunks</span>
                </div>
                <div class="hero-stat-sep"></div>
                <div class="hero-stat-box">
                    <span class="stat-number">BGE-Large</span>
                    <span class="stat-label">1024-D Dense Vectors</span>
                </div>
                <div class="hero-stat-sep"></div>
                <div class="hero-stat-box">
                    <span class="stat-number">21-Rule</span>
                    <span class="stat-label">Legal Verifier & Citations</span>
                </div>
                <div class="hero-stat-sep"></div>
                <div class="hero-stat-box">
                    <span class="stat-number">Neural TTS</span>
                    <span class="stat-label">Indian English Accent</span>
                </div>
            </div>
        </div>
        """)
    
    with gr.Row():
        # LEFT COLUMN: QUERY CONSOLE
        with gr.Column(scale=5, elem_classes="query-panel"):
            gr.HTML("""
            <div class="panel-header-custom">
                <span>Query Console & Statutory Framing</span>
            </div>
            """)

            query_box = gr.Textbox(
                lines=3,
                label="Enter Legal Question / Statutory Issue",
                placeholder="e.g. What does Section 9A of the Representation of the People Act state regarding disqualification for government contracts?"
            )

            # Quick Prompt Chips for immediate 1-click test driving
            with gr.Group(elem_classes="sample-chips-wrap"):
                gr.HTML('<div class="chips-label">Quick Landmark Legal Inquiries:</div>')
                with gr.Row():
                    chip_btns = []
                    for item in SAMPLE_QUESTIONS:
                        btn = gr.Button(item["label"], size="sm", elem_classes="sample-chip-btn")
                        chip_btns.append((btn, item["query"]))

            # Dynamic AI Classifier Card
            detected_area = gr.HTML(initial_detected_html)
            detected_domain = gr.Textbox(value=default_domain, visible=False)
            detected_subdomain = gr.Textbox(value=default_subs[0] if default_subs else "Fundamental Rights", visible=False)

            # Wire query box typing to update classification
            query_box.input(
                update_detected_area,
                inputs=[query_box],
                outputs=[detected_area, detected_domain, detected_subdomain]
            )

            # Wire sample chip buttons to set query and update classification
            for btn, sample_q in chip_btns:
                btn.click(
                    fn=lambda q=sample_q: (q, *update_detected_area(q)),
                    inputs=[],
                    outputs=[query_box, detected_area, detected_domain, detected_subdomain]
                )

            advanced_mode = gr.Checkbox(label="Advanced: Override legal domain taxonomy manually", value=False)
            with gr.Group(visible=False) as manual_group:
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

            advanced_mode.change(
                lambda enabled: gr.update(visible=enabled),
                inputs=[advanced_mode],
                outputs=[manual_group]
            )

            voice_choice = gr.Radio(
                choices=[("Female (Neerja Neural)", "Female"), ("Male (Prabhat Neural)", "Male")],
                value="Female",
                label="Neural TTS Voice (Indian English Accent)"
            )
            submit_btn = gr.Button("Execute Grounded Legal Search & Voice", variant="primary", elem_classes="primary-search-btn")
            
        # RIGHT COLUMN: VERIFIED LEGAL ANSWERS
        with gr.Column(scale=6, elem_classes="answer-panel"):
            gr.HTML("""
            <div class="panel-header-custom">
                <span>Verified Legal Findings & Oral Synthesis</span>
            </div>
            """)

            answer_box = gr.Textbox(
                lines=8,
                label="Verified Statutory Answer",
                placeholder="The grounded legal answer with statutory citations will be generated here..."
            )
            explain_btn = gr.Button("Explain in Simple Words (Client Advisory)", interactive=False, elem_classes="explain-action-btn")
            simple_explain_box = gr.Markdown(visible=False, elem_classes="simple-explain-card")
            citations_box = gr.Markdown(label="Statutory Citations & Vector Retrieval Origin", elem_classes="citations-card")
            audio_box = gr.Audio(label="Spoken Neural Voice Explanation (Indian English)", type="filepath", elem_classes="audio-card")

    submit_btn.click(
        handle_gradio_query,
        inputs=[query_box, detected_domain, detected_subdomain, advanced_mode, domain_dropdown, subdomain_dropdown, voice_choice],
        outputs=[answer_box, citations_box, audio_box, explain_btn, simple_explain_box]
    )

    explain_btn.click(
        handle_explain_simple,
        inputs=[answer_box, citations_box],
        outputs=[simple_explain_box]
    )

# Launch the Gradio app directly with show_api=False to prevent JSON schema parse errors
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

