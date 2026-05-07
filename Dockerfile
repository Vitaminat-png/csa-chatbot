# ============================================================
# CSA AI Chatbot — Dockerfile
# Target: Render.com (free tier, amd64)
# ============================================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (docs/ and .env excluded via .dockerignore)
COPY . .

# Expose the port uvicorn will listen on
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
