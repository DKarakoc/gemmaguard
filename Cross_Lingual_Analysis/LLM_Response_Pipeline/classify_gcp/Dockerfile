# Dockerfile for Vertex AI Custom Job
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies including gcloud CLI
RUN apt-get update && apt-get install -y \
    git \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    transformers \
    peft \
    datasets \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    scikit-learn \
    google-cloud-storage \
    gcsfs \
    python-dotenv \
    tqdm

ARG KEY_FILE
COPY classify_script.py /app/classify.py
COPY ${KEY_FILE} /app/service-account-key.json

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

# Entry point - run the classification script
ENTRYPOINT ["python", "classify.py"]
