FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace model into the image to eliminate startup delays
RUN python -c "from transformers import AutoTokenizer, EsmForMaskedLM; AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D'); EsmForMaskedLM.from_pretrained('facebook/esm2_t12_35M_UR50D')"

# Copy application files
COPY . .

# Environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run using gunicorn for production. Extended timeout for initial edge-compute warm-up
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "run:app"]
