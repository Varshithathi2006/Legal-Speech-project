FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files and vectorstore
COPY --chown=user app.py .
COPY --chown=user legal_rag_qa.py .
COPY --chown=user legal_system_prompt.txt .
COPY --chown=user legal_rag_vectorstore/ ./legal_rag_vectorstore/

# Create output audio dir for TTS caching
RUN mkdir -p output_audio

# Hugging Face Space Streamlit Configuration
EXPOSE 7860
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
