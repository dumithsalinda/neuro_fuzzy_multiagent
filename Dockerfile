# Dockerfile for Neuro-Fuzzy Multi-Agent System Dashboard & API
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Default command: run Streamlit dashboard
CMD ["streamlit", "run", "dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
