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

def handle_gradio_query(query, domain, subdomain, voice_gender):
    if not query or not query.strip():
        return "Please enter a legal question.", "", None
    
    try:
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
            domain_dropdown = gr.Dropdown(
                choices=list(LEGAL_DOMAINS.keys()),
                value=default_domain,
                label="Step 1: Primary Legal Domain"
            )
            subdomain_dropdown = gr.Dropdown(
                choices=all_subdomains,
                value=default_subs[0] if default_subs else None,
                label="Step 2: Legal Subdomain"
            )
            voice_choice = gr.Radio(
                choices=["Female", "Male"],
                value="Female",
                label="Neural TTS Voice (Indian English)"
            )
            
            def on_domain_change(d):
                subs = LEGAL_DOMAINS.get(d, [])
                return gr.update(choices=subs, value=subs[0] if subs else None)
            
            domain_dropdown.change(on_domain_change, inputs=[domain_dropdown], outputs=[subdomain_dropdown])
            
            query_box = gr.Textbox(
                lines=3,
                label="Enter Legal Question",
                placeholder="e.g. What does Section 9A of the Representation of the People Act state regarding disqualification for government contracts?"
            )
            submit_btn = gr.Button("🔍 Execute Legal Search & Voice", variant="primary", elem_classes="primary-action")
            
        with gr.Column(scale=6, elem_classes="answer-panel"):
            answer_box = gr.Textbox(lines=7, label="Verified Legal Answer")
            citations_box = gr.Textbox(lines=3, label="Statutory & Case Citations")
            audio_box = gr.Audio(label="Spoken Neural Voice Explanation", type="filepath")

    submit_btn.click(
        handle_gradio_query,
        inputs=[query_box, domain_dropdown, subdomain_dropdown, voice_choice],
        outputs=[answer_box, citations_box, audio_box]
    )

# Launch the Gradio app directly with show_api=False to prevent JSON schema parse errors
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
