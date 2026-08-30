import os
import sys
import json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from api import app
from legal_rag_qa import (
    process_hierarchical_legal_query,
    LEGAL_DOMAINS,
    SUBDOMAIN_APPLICABLE_LAW
)

def handle_gradio_query(query, domain, subdomain, voice_gender):
    if not query.strip():
        return "Please enter a legal question.", "", None
    ans, cites, aud_path, metrics = process_hierarchical_legal_query(
        query_text=query,
        domain=domain,
        subdomain=subdomain,
        voice_gender=voice_gender
    )
    return ans, cites, aud_path

# Create Gradio Blocks Interface
with gr.Blocks(title="Indian Legal Speech RAG Studio & API", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# ⚖️ Indian Legal Speech RAG Studio & API")
    gr.Markdown("**Grounded Spoken Question Answering across Courtroom Proceedings and Indian Statutes**")
    
    with gr.Row():
        with gr.Column(scale=1):
            domain_dropdown = gr.Dropdown(
                choices=list(LEGAL_DOMAINS.keys()),
                value="Constitutional & Administrative Law",
                label="Step 1: Primary Legal Domain"
            )
            subdomain_dropdown = gr.Dropdown(
                choices=LEGAL_DOMAINS["Constitutional & Administrative Law"],
                value="Election & Representation Law",
                label="Step 2: Legal Subdomain"
            )
            voice_choice = gr.Radio(
                choices=["Female", "Male"],
                value="Female",
                label="Neural TTS Voice (Indian English)"
            )
            
            def update_subdomains(d):
                subs = LEGAL_DOMAINS.get(d, [])
                return gr.Dropdown(choices=subs, value=subs[0] if subs else None)
            
            domain_dropdown.change(update_subdomains, inputs=[domain_dropdown], outputs=[subdomain_dropdown])
            
            query_box = gr.Textbox(
                lines=3,
                label="Enter Legal Question",
                placeholder="e.g. What does Section 9A of the Representation of the People Act state regarding disqualification for government contracts?"
            )
            submit_btn = gr.Button("Execute Legal Search & Voice", variant="primary")
            
            gr.Examples(
                examples=[
                    ["What does Section 9A of the Representation of the People Act, 1951 state regarding disqualification for government contracts?", "Constitutional & Administrative Law", "Election & Representation Law", "Female"],
                    ["What are the statutory grounds for setting aside an arbitral award under Section 34 of the Arbitration and Conciliation Act?", "Corporate & Business Law", "Arbitration & Dispute Resolution", "Female"],
                    ["What are the statutory conditions for granting bail in a non-bailable offence under Section 437 of the CrPC?", "Criminal Law", "Bail & Criminal Procedure", "Female"],
                    ["How is the right to life and personal liberty protected under Article 21 of the Indian Constitution?", "Constitutional & Administrative Law", "Fundamental Rights & Writs", "Female"]
                ],
                inputs=[query_box, domain_dropdown, subdomain_dropdown, voice_choice]
            )

        with gr.Column(scale=1):
            answer_box = gr.Textbox(lines=6, label="Verified Legal Answer")
            citations_box = gr.Textbox(lines=3, label="Statutory & Transcript Citations")
            audio_box = gr.Audio(label="Spoken Neural Voice Explanation", type="filepath")

    submit_btn.click(
        handle_gradio_query,
        inputs=[query_box, domain_dropdown, subdomain_dropdown, voice_choice],
        outputs=[answer_box, citations_box, audio_box]
    )

# Mount Gradio app onto FastAPI app so both UI and REST endpoints (/api/query, /docs) work simultaneously
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
