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
<div class="router-card" style="background:#F8FAFC; border:1px solid #D1D9E2; border-left:4px solid #2563EB; border-radius:8px; padding:12px 16px; margin:14px 0;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:11.5px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#52647A;">Detected Legal Domain</span>
        <span style="background:#EFF6FF; border:1px solid #BFDBFE; color:#1E40AF; font-size:12px; font-weight:600; padding:2px 8px; border-radius:4px;">{domain}</span>
    </div>
    <div style="font-size:14.5px; color:#17243A; margin-bottom:8px; line-height:1.4;">
        <span style="color:#52647A; font-size:12.5px; font-weight:500;">Classification:</span>
        <strong style="color:#17243A; margin-left:4px;">{domain} &rsaquo; {subdomain}</strong>
    </div>
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:8px 12px; font-size:13.5px; color:#17243A; line-height:1.4;">
        <span style="color:#52647A; font-weight:500;">Governing Legislation:</span>
        <strong style="color:#17243A; margin-left:4px;">{applicable_law}</strong>
    </div>
</div>
"""


def format_applicable_law_card(domain, subdomain):
    applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(subdomain, subdomain)
    return f"""
<div class="law-context-card" style="background:#F8FAFC; border:1px solid #D1D9E2; border-radius:8px; padding:10px 14px; margin-bottom:12px; display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
    <div style="display:flex; flex-direction:column;">
        <span style="font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#52647A; font-weight:600;">Enactment</span>
        <span style="font-size:13.5px; font-weight:600; color:#17243A;">{applicable_law}</span>
    </div>
    <div style="width:1px; height:18px; background:#D1D9E2;"></div>
    <div style="display:flex; flex-direction:column;">
        <span style="font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#52647A; font-weight:600;">Domain</span>
        <span style="font-size:13.5px; font-weight:600; color:#17243A;">{domain} &rsaquo; {subdomain}</span>
    </div>
    <div style="width:1px; height:18px; background:#D1D9E2;"></div>
    <div style="display:flex; flex-direction:column;">
        <span style="font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#52647A; font-weight:600;">Jurisdiction</span>
        <span style="font-size:13.5px; font-weight:600; color:#17243A;">Republic of India</span>
    </div>
</div>
"""


def update_detected_area(query):
    domain, subdomain = classify_legal_question(query)
    card_html = format_detected_area_html(domain, subdomain)
    return card_html, domain, subdomain


def clean_citation_text(raw_cites):
    if not raw_cites or not str(raw_cites).strip():
        return ""
    text = str(raw_cites).strip()
    
    # Strip any and all unicode emojis (e.g. pin, scroll, mic, lightning)
    emoji_pattern = re.compile(
        r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\ufe00-\ufe0f]',
        flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub('', text)
    cleaned = cleaned.replace("**Retrieved From:**", "").replace("Retrieved From:", "").strip()
    
    # Convert markdown bold and italic to inline HTML with explicit dark navy styling
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#17243A !important;">\1</strong>', cleaned)
    cleaned = re.sub(r'\*(.*?)\*', r'<em style="color:#2D3748 !important;">\1</em>', cleaned)
    
    if "\n" in cleaned:
        items = [line.strip().lstrip("- ").strip() for line in cleaned.split("\n") if line.strip()]
        rendered_items = "".join([f'<li style="color:#17243A !important; margin-bottom:5px; font-size:13.5px; line-height:1.5;">{it}</li>' for it in items])
        body_html = f'<ul style="margin:6px 0 0 16px; padding:0; color:#17243A !important;">{rendered_items}</ul>'
    else:
        body_html = f'<div style="color:#17243A !important; font-size:13.5px; font-weight:500; margin-top:4px; line-height:1.5;">{cleaned}</div>'
    
    return f"""
<div class="citations-card" style="background:#F8FAFC !important; border:1px solid #D1D9E2 !important; border-left:3px solid #2563EB !important; border-radius:8px !important; padding:12px 16px !important; margin-top:10px !important; margin-bottom:12px !important;">
    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#52647A !important; font-weight:700;">Supporting Evidence & Provenance</div>
    {body_html}
</div>
"""


def handle_gradio_query(query, detected_domain, detected_subdomain, manual_mode, manual_domain, manual_subdomain, voice_gender):
    if not query or not query.strip():
        empty_law = format_applicable_law_card("Constitutional & Administrative Law", "Fundamental Rights")
        return "Please enter a legal question to initiate statutory analysis.", empty_law, "", gr.update(value=None, visible=False), gr.update(interactive=False), gr.update(value="", visible=False)
    
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
        clean_cites_html = clean_citation_text(cites)
        
        audio_update = gr.update(value=aud_path, visible=True) if (aud_path and os.path.exists(aud_path)) else gr.update(value=None, visible=False)
        
        return ans, law_card_html, clean_cites_html, audio_update, gr.update(interactive=is_valid), gr.update(value="", visible=False)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        error_msg = f"**An error occurred during query processing:** {str(e)}\n\nPlease try rephrasing your legal question or selecting the domain again."
        empty_law = format_applicable_law_card(detected_domain, detected_subdomain)
        return error_msg, empty_law, "", gr.update(value=None, visible=False), gr.update(interactive=False), gr.update(value="", visible=False)


def handle_explain_simple(verified_answer, citations):
    if not verified_answer or not verified_answer.strip() or verified_answer.startswith("Please enter") or verified_answer.startswith("Error") or verified_answer.startswith("\u26a0"):
        return gr.update(value="<div style='color:#52647A; font-size:13.5px; padding:10px;'>Generate a verified legal answer first to view the client-friendly explanation.</div>", visible=True)
    
    raw_citations_for_prompt = re.sub(r'<[^>]+>', ' ', str(citations)).strip() if citations else ""
    explanation = explain_legal_answer_simple(verified_answer, raw_citations_for_prompt)
    
    clean_exp = explanation
    clean_exp = re.sub(r'###?\s*In Simple Words', '#### 1. Core Meaning', clean_exp, flags=re.IGNORECASE)
    clean_exp = re.sub(r'###?\s*Simple Example', '#### 2. Real-World Application', clean_exp, flags=re.IGNORECASE)
    clean_exp = re.sub(r'###?\s*Important Point', '#### 3. Practical Takeaway', clean_exp, flags=re.IGNORECASE)
    
    if '#### 1. Core Meaning' not in clean_exp:
        formatted = f"""### Plain-Language Client Advisory

#### 1. Core Meaning
{clean_exp}

#### 2. Real-World Application
This statutory rule governs rights and obligations in commercial dealings, regulatory compliance, and courtroom proceedings under Indian law.

#### 3. Practical Takeaway
Parties and counsel must ensure compliance with all procedural preconditions and evidentiary standards established by the governing statute."""
    else:
        formatted = f"""### Plain-Language Client Advisory\n\n{clean_exp}"""
    return gr.update(value=formatted, visible=True)


# Default subdomains for initial load
default_domain = "Constitutional & Administrative Law"
default_subs = LEGAL_DOMAINS.get(default_domain, ["Fundamental Rights"])
all_subdomains = sorted(list({s for subs in LEGAL_DOMAINS.values() for s in subs}))
initial_detected_html = format_detected_area_html(default_domain, default_subs[0] if default_subs else "Fundamental Rights")
initial_law_card_html = format_applicable_law_card(default_domain, default_subs[0] if default_subs else "Fundamental Rights")

# Sample questions matching exact professional labels requested
SAMPLE_QUESTIONS = [
    {
        "label": "Section 9A — Representation of the People Act",
        "query": "What does Section 9A of the Representation of the People Act, 1951 state regarding disqualification for government contracts?"
    },
    {
        "label": "Article 21 — Right to Privacy",
        "query": "How is the right to life and personal liberty protected under Article 21 of the Constitution of India in Supreme Court jurisprudence?"
    },
    {
        "label": "Section 65B — Electronic Evidence",
        "query": "What are the mandatory certificate requirements for electronic records admissibility under Section 65B of the Indian Evidence Act, 1872?"
    },
    {
        "label": "Section 437 — Bail under the CrPC",
        "query": "What statutory conditions and judicial principles govern the grant of bail under Section 437 of the Code of Criminal Procedure, 1973 for non-bailable offences?"
    },
    {
        "label": "Section 34 — Setting Aside Arbitral Awards",
        "query": "On what statutory grounds can an arbitral award be set aside under Section 34 of the Arbitration and Conciliation Act, 1996?"
    }
]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base: #F8FAFC;
    --bg-surface: #FFFFFF;
    --border-subtle: #D1D9E2;
    --text-primary: #17243A;
    --text-secondary: #52647A;
    --text-muted: #718294;
    --accent-orange: #C65D21;
    --accent-orange-hover: #A84A15;
    --accent-blue: #2563EB;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container, .dark, .dark body, .dark .gradio-container {
    background-color: #F8FAFC !important;
    color: #17243A !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px 24px 48px !important;
}

/* ==========================================================================
   Header and System Specification Bar (Clean white card, no dark strip)
   ========================================================================== */
.studio-hero {
    background: #FFFFFF !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 1px 3px rgba(23, 36, 58, 0.04) !important;
}

.hero-header-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    background: transparent !important;
}

.brand-identity {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
    background: transparent !important;
}

.brand-icon-svg {
    width: 40px !important;
    height: 40px !important;
    background: #FBF6F2 !important;
    border: 1px solid #EAD8C7 !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
    color: #C65D21 !important;
}

.brand-title-wrap {
    display: flex !important;
    flex-direction: column !important;
    background: transparent !important;
}

.brand-title {
    font-size: 21px !important;
    font-weight: 700 !important;
    color: #17243A !important;
    margin: 0 !important;
    line-height: 1.25 !important;
}

.brand-subtitle {
    font-size: 13.5px !important;
    color: #52647A !important;
    margin: 3px 0 0 0 !important;
    line-height: 1.4 !important;
}

/* System Capabilities Grid (clearly labeled as configured specs) */
.capabilities-bar {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)) !important;
    gap: 12px !important;
    padding-top: 16px !important;
    margin-top: 16px !important;
    border-top: 1px solid #E2E8F0 !important;
}

.capability-item {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    display: flex !important;
    flex-direction: column !important;
}

.capability-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #17243A !important;
}

.capability-lbl {
    font-size: 12px !important;
    color: #52647A !important;
    margin-top: 2px !important;
}

/* ==========================================================================
   Panel Containers
   ========================================================================== */
.query-panel, .answer-panel {
    background: #FFFFFF !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 12px !important;
    padding: 22px !important;
    box-shadow: 0 1px 4px rgba(23, 36, 58, 0.04) !important;
}

.section-title-wrap {
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}

.section-title {
    font-size: 19px !important;
    font-weight: 700 !important;
    color: #17243A !important;
    margin: 0 0 4px 0 !important;
    line-height: 1.3 !important;
}

.section-desc {
    font-size: 13.5px !important;
    color: #52647A !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

/* ==========================================================================
   Input Textareas and Labels
   ========================================================================== */
.gradio-container textarea, .gradio-container input[type="text"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 8px !important;
    color: #17243A !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    padding: 12px 14px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: #C65D21 !important;
    box-shadow: 0 0 0 3px rgba(198, 93, 33, 0.15) !important;
    outline: none !important;
}

.gradio-container textarea::placeholder {
    color: #718294 !important;
    font-size: 13.5px !important;
}

.gradio-container label span, .label-wrap span {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #17243A !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}

/* Primary Action Button */
.primary-action-btn {
    min-height: 44px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    background: #C65D21 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(198, 93, 33, 0.22) !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
    cursor: pointer !important;
    margin-top: 10px !important;
}

.primary-action-btn:hover {
    background: #A84A15 !important;
    transform: translateY(-1px) !important;
}

.primary-action-btn:active {
    transform: translateY(0) !important;
}

/* Suggested Legal Queries Section */
.suggested-queries-title {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #17243A !important;
    margin: 16px 0 8px 0 !important;
}

.suggested-queries-container {
    gap: 6px !important;
    margin-bottom: 12px !important;
}

.suggested-query-btn {
    width: 100% !important;
    min-height: 38px !important;
    background: #F8FAFC !important;
    border: 1px solid #D1D9E2 !important;
    color: #17243A !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    white-space: normal !important;
    line-height: 1.4 !important;
}

.suggested-query-btn:hover {
    background: #EEF2F6 !important;
    border-color: #B8C5D3 !important;
    color: #17243A !important;
}

.suggested-query-btn:focus {
    border-color: #C65D21 !important;
    outline: none !important;
}

/* ==========================================================================
   Answer and Evidence Workspace
   ========================================================================== */
.answer-panel textarea {
    font-size: 15px !important;
    line-height: 1.55 !important;
}

.explain-action-btn {
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    min-height: 38px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    background: #FFFFFF !important;
    color: #17243A !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.explain-action-btn:enabled:hover {
    background: #F8FAFC !important;
    border-color: #C65D21 !important;
    color: #C65D21 !important;
}

.simple-explain-card {
    background: #FBF6F2 !important;
    border: 1px solid #EAD8C7 !important;
    border-left: 3px solid #C65D21 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    font-size: 14.5px !important;
    line-height: 1.55 !important;
    color: #17243A !important;
}

.simple-explain-card h3, .simple-explain-card h4 {
    color: #17243A !important;
    margin: 10px 0 4px 0 !important;
}

.simple-explain-card h4 {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #C65D21 !important;
}

.citations-card, .dark .citations-card {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    border: 1px solid #D1D9E2 !important;
    border-left: 3px solid #2563EB !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin-top: 10px !important;
    margin-bottom: 12px !important;
    font-size: 13.5px !important;
    color: #17243A !important;
    line-height: 1.5 !important;
}

.citations-card *, .dark .citations-card *,
.citations-container *, .dark .citations-container * {
    color: #17243A !important;
}

/* ==========================================================================
   Accordion - High Contrast Light Surface
   ========================================================================== */
.accordion, .dark .accordion,
details, .dark details,
.gradio-accordion, .dark .gradio-accordion,
details.accordion, .dark details.accordion {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 8px !important;
    margin-top: 14px !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 2px rgba(23, 36, 58, 0.03) !important;
}

details summary, .dark details summary,
.accordion > .label-wrap, .dark .accordion > .label-wrap,
summary.label-wrap, .dark summary.label-wrap {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    border: none !important;
    border-bottom: 1px solid #E2E8F0 !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    color: #17243A !important;
}

details summary:hover, .dark details summary:hover {
    background: #EEF2F6 !important;
    background-color: #EEF2F6 !important;
}

details summary *, .dark details summary *,
details summary span, .dark details summary span,
.accordion summary *, .dark .accordion summary *,
.accordion .label-wrap *, .dark .accordion .label-wrap * {
    color: #17243A !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}

details > div, .dark details > div,
details[open] > div, .dark details[open] > div {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    padding: 14px !important;
}

/* ==========================================================================
   Radio Group & Form Fields (Full Dark Mode Override)
   ========================================================================== */
.gradio-radio, .dark .gradio-radio,
fieldset.gradio-radio, .dark fieldset.gradio-radio,
.block.gradio-radio, .dark .block.gradio-radio {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    margin-top: 14px !important;
    box-shadow: 0 1px 2px rgba(23, 36, 58, 0.03) !important;
}

.gradio-radio > .label-wrap, .dark .gradio-radio > .label-wrap,
.gradio-radio .label-wrap span, .dark .gradio-radio .label-wrap span,
.gradio-radio > label, .dark .gradio-radio > label,
.gradio-radio > label > span, .dark .gradio-radio > label > span {
    background: transparent !important;
    background-color: transparent !important;
    color: #17243A !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
    display: inline-block !important;
}

.gradio-radio .wrap, .dark .gradio-radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
}

.gradio-radio label, .dark .gradio-radio label,
.gradio-radio .wrap > label, .dark .gradio-radio .wrap > label {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    flex: 1 1 200px !important;
}

.gradio-radio label:hover, .dark .gradio-radio label:hover {
    background: #EEF2F6 !important;
    border-color: #CBD5E1 !important;
}

.gradio-radio label.selected, .dark .gradio-radio label.selected,
.gradio-radio label:has(input:checked), .dark .gradio-radio label:has(input:checked) {
    background: #FFF7ED !important;
    background-color: #FFF7ED !important;
    border: 1.5px solid #C65D21 !important;
    box-shadow: 0 0 0 1px #C65D21 !important;
}

.gradio-radio label span, .dark .gradio-radio label span,
.gradio-radio label input + span, .dark .gradio-radio label input + span {
    color: #17243A !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
}

.gradio-radio label.selected span, .dark .gradio-radio label.selected span,
.gradio-radio label:has(input:checked) span, .dark .gradio-radio label:has(input:checked) span {
    color: #C65D21 !important;
    font-weight: 600 !important;
}

.gradio-radio input[type="radio"], .dark .gradio-radio input[type="radio"] {
    accent-color: #C65D21 !important;
    width: 15px !important;
    height: 15px !important;
}

/* ==========================================================================
   Audio Card (Clean styling, no orange pill badges)
   ========================================================================== */
.audio-card, .dark .audio-card {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #D1D9E2 !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    margin-top: 10px !important;
    box-shadow: 0 1px 2px rgba(23, 36, 58, 0.03) !important;
}

.audio-card .label-wrap, .dark .audio-card .label-wrap,
.audio-card label span, .dark .audio-card label span {
    background: transparent !important;
    background-color: transparent !important;
    color: #17243A !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 0 8px 0 !important;
    margin: 0 !important;
    display: block !important;
}

/* Neutralize label badge pills */
.label-wrap, .dark .label-wrap,
span.has-info, .dark span.has-info,
.block > label, .dark .block > label,
label > span, .dark label > span,
.gr-form > label, .dark .gr-form > label {
    background: transparent !important;
    background-color: transparent !important;
    color: #17243A !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* Dropdown & Checkbox overrides */
.gradio-dropdown, .dark .gradio-dropdown,
.gradio-dropdown select, .dark .gradio-dropdown select {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border-color: #D1D9E2 !important;
    color: #17243A !important;
}

.gradio-checkbox label, .dark .gradio-checkbox label {
    background: transparent !important;
    color: #17243A !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
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

# Client script to permanently prevent and remove dark mode injection
js_light_enforcer = """
() => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    const observer = new MutationObserver(() => {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
        }
        if (document.body.classList.contains('dark')) {
            document.body.classList.remove('dark');
        }
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
}
"""

theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#F8FAFC",
    body_background_fill_dark="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#D1D9E2",
    block_border_color_dark="#D1D9E2",
    body_text_color="#17243A",
    body_text_color_dark="#17243A",
    body_text_color_subdued="#52647A",
    body_text_color_subdued_dark="#52647A",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#17243A",
    block_label_text_color_dark="#17243A",
    block_title_text_color="#17243A",
    block_title_text_color_dark="#17243A",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#D1D9E2",
    input_border_color_dark="#D1D9E2",
    checkbox_label_background_fill="#FFFFFF",
    checkbox_label_background_fill_dark="#FFFFFF",
    checkbox_label_text_color="#17243A",
    checkbox_label_text_color_dark="#17243A",
    accordion_text_color="#17243A",
    accordion_text_color_dark="#17243A",
)

# Create Gradio Interface
with gr.Blocks(title="Indian Legal Speech RAG Studio", theme=theme, css=custom_css, js=js_light_enforcer) as demo:
    # -- HEADER & CONFIGURED ARCHITECTURE BAR (Clean card, no dark strip) --
    gr.HTML("""
    <div class="studio-hero">
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
            <div class="section-title-wrap">
                <h3 class="section-title">Ask a Legal Question</h3>
                <p class="section-desc">Enter a statutory question or describe the legal issue you want to understand.</p>
            </div>
            """)

            query_box = gr.Textbox(
                lines=4,
                show_label=False,
                placeholder="Example: Under Section 34 of the Arbitration and Conciliation Act, 1996, what are the grounds for setting aside an arbitral award?"
            )

            submit_btn = gr.Button("Analyze Legal Query", variant="primary", elem_classes="primary-action-btn")

            # Suggested Legal Queries Section
            gr.HTML('<div class="suggested-queries-title">Suggested Legal Queries</div>')
            with gr.Column(elem_classes="suggested-queries-container"):
                chip_btns = []
                for item in SAMPLE_QUESTIONS:
                    btn = gr.Button(item["label"], elem_classes="suggested-query-btn")
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
                choices=[("Female — Neerja (Indian English)", "Female"), ("Male — Prabhat (Indian English)", "Male")],
                value="Female",
                label="Spoken Narration Voice"
            )

        # -- RIGHT COLUMN: LEGAL ANSWER & EVIDENCE WORKSPACE --
        with gr.Column(scale=6, elem_classes="answer-panel"):
            gr.HTML("""
            <div class="section-title-wrap">
                <h3 class="section-title">Legal Analysis</h3>
            </div>
            """)

            # Section 2: Applicable Law Banner
            applicable_law_card = gr.HTML(initial_law_card_html)

            # Section 1: Grounded Legal Answer
            answer_box = gr.Textbox(
                lines=9,
                show_label=False,
                placeholder="The grounded statutory analysis will appear here once the query is processed..."
            )

            # Section 5: Client Advisory (Plain-Language Explanation)
            explain_btn = gr.Button("Explain in Simple Words (Client Advisory)", interactive=False, elem_classes="explain-action-btn")
            simple_explain_box = gr.Markdown(visible=False, elem_classes="simple-explain-card")

            # Section 3 & 4: Supporting Evidence, Citations & Provenance
            citations_box = gr.HTML(value="", elem_classes="citations-container")

            # Section 6: Spoken Neural Voice Output (Initially hidden until generated)
            audio_box = gr.Audio(label="Spoken Neural Voice Explanation (Indian English)", type="filepath", elem_classes="audio-card", visible=False)

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


