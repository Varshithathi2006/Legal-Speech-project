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


def update_detected_area(query):
    domain, subdomain = classify_legal_question(query)
    applicable_law = SUBDOMAIN_APPLICABLE_LAW.get(subdomain, subdomain)
    area = f"**Detected area:** {domain} -> {subdomain}\n\n*Applicable law: {applicable_law}*"
    return area, domain, subdomain


def handle_gradio_query(query, detected_domain, detected_subdomain, manual_mode, manual_domain, manual_subdomain, voice_gender):
    if not query or not query.strip():
        return "Please enter a legal question.", "", None
    
    try:
        domain = manual_domain if manual_mode else detected_domain
        subdomain = manual_subdomain if manual_mode else detected_subdomain
        # Run the 21-rule RAG retrieval & synthesis pipeline
        ans, cites, aud_path, metrics = process_hierarchical_legal_query(
            query_text=query.strip(),
            domain=domain,
            subdomain=subdomain,
            voice_gender=voice_gender or "Female"
        )
        return ans, cites, aud_path
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        error_msg = f"⚠️ **An error occurred during query processing:** {str(e)}\n\nPlease try rephrasing your legal question or selecting the domain again."
        return error_msg, "N/A - Execution Error", None

# Default subdomains for initial load
default_domain = "Constitutional & Administrative Law"
default_subs = LEGAL_DOMAINS.get(default_domain, ["Fundamental Rights"])
all_subdomains = sorted(list({s for subs in LEGAL_DOMAINS.values() for s in subs}))

custom_css = """
.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 28px 34px 42px !important;
}

.app-header {
    margin-bottom: 22px;
}

.app-header h1 {
    font-size: clamp(1.8rem, 3vw, 2.8rem) !important;
    letter-spacing: 0 !important;
    margin-bottom: 8px !important;
}

.app-header p {
    color: #b8c4d8 !important;
    font-size: 1.05rem !important;
}

.query-panel, .answer-panel {
    border: 1px solid #2d3b54 !important;
    border-radius: 14px !important;
    padding: 22px !important;
    background: #172238 !important;
}

.answer-panel textarea {
    font-size: 1rem !important;
    line-height: 1.55 !important;
}

.primary-action {
    margin-top: 12px !important;
    min-height: 52px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

@media (max-width: 760px) {
    .gradio-container {
        padding: 18px 14px 28px !important;
    }

    .query-panel, .answer-panel {
        padding: 16px !important;
    }
}
"""

# Create Gradio Interface
with gr.Blocks(title="Indian Legal Speech RAG Studio & API", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    with gr.Group(elem_classes="app-header"):
        gr.Markdown("# ⚖️ Indian Legal Speech RAG Studio")
        gr.Markdown("Grounded spoken answers from Indian statutes and courtroom proceedings")
    
    with gr.Row():
        with gr.Column(scale=5, elem_classes="query-panel"):
            query_box = gr.Textbox(
                lines=3,
                label="Enter Legal Question",
                placeholder="e.g. What does Section 9A of the Representation of the People Act state regarding disqualification for government contracts?"
            )
            detected_area = gr.Markdown(
                "**Detected area:** Constitutional & Administrative Law -> Fundamental Rights\n\n*Applicable law: Constitution of India - Part III (Articles 12-35)*"
            )
            detected_domain = gr.Textbox(value=default_domain, visible=False)
            detected_subdomain = gr.Textbox(value=default_subs[0] if default_subs else "Fundamental Rights", visible=False)
            query_box.input(
                update_detected_area,
                inputs=[query_box],
                outputs=[detected_area, detected_domain, detected_subdomain]
            )

            advanced_mode = gr.Checkbox(label="Advanced: choose the legal category manually", value=False)
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
                choices=["Female", "Male"],
                value="Female",
                label="Neural TTS Voice (Indian English)"
            )
            submit_btn = gr.Button("🔍 Execute Legal Search & Voice", variant="primary", elem_classes="primary-action")
            
        with gr.Column(scale=6, elem_classes="answer-panel"):
            answer_box = gr.Textbox(lines=7, label="Verified Legal Answer")
            citations_box = gr.Textbox(lines=3, label="Statutory & Case Citations")
            audio_box = gr.Audio(label="Spoken Neural Voice Explanation", type="filepath")

    submit_btn.click(
        handle_gradio_query,
        inputs=[query_box, detected_domain, detected_subdomain, advanced_mode, domain_dropdown, subdomain_dropdown, voice_choice],
        outputs=[answer_box, citations_box, audio_box]
    )

# Launch the Gradio app directly with show_api=False to prevent JSON schema parse errors
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
