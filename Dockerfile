FROM python:3.9-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set SPACY_DATA environment variable and download spaCy models to a persistent location
ENV SPACY_DATA /usr/local/spacy_data
RUN mkdir -p ${SPACY_DATA} \
    && python -m spacy download en_core_web_sm --target ${SPACY_DATA}

# Set NLTK_DATA environment variable and download NLTK data to a persistent location
ENV NLTK_DATA /usr/local/nltk_data
RUN mkdir -p ${NLTK_DATA} \
    && python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}'); nltk.download('stopwords', download_dir='${NLTK_DATA}'); nltk.download('punkt_tab', download_dir='${NLTK_DATA}')"

FROM python:3.9-slim-bookworm AS runner

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/nltk_data /usr/local/nltk_data
COPY --from=builder /usr/local/spacy_data /usr/local/spacy_data

COPY . .

ENV PYTHONPATH=/app
ENV NLTK_DATA /usr/local/nltk_data
ENV SPACY_DATA /usr/local/spacy_data

CMD ["python", "main.py"]


