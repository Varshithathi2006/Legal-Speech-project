import os
import sys
import json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

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

# Create Gradio Interface
with gr.Blocks(title="Indian Legal Speech RAG Studio & API", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# ⚖️ Indian Legal Speech RAG Studio")
    gr.Markdown("### Grounded Spoken AI across Courtroom Proceedings & Indian Statutes")
    
    with gr.Row():
        with gr.Column(scale=1):
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
            submit_btn = gr.Button("🔍 Execute Legal Search & Voice", variant="primary")
            
            gr.Examples(
                examples=[
                    ["What does Section 9A of the Representation of the People Act, 1951 state regarding disqualification for government contracts?", "Constitutional & Administrative Law", "Fundamental Rights", "Female"],
                    ["What are the statutory grounds for setting aside an arbitral award under Section 34 of the Arbitration and Conciliation Act?", "Corporate & Business Law", "Contract Law", "Female"],
                    ["What are the statutory conditions for granting bail in a non-bailable offence under Section 437 of the CrPC?", "Criminal Law", "Bail Procedures", "Female"],
                    ["How is the right to life and personal liberty protected under Article 21 of the Indian Constitution?", "Constitutional & Administrative Law", "Fundamental Rights", "Female"]
                ],
                inputs=[query_box, domain_dropdown, subdomain_dropdown, voice_choice]
            )

        with gr.Column(scale=1):
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
