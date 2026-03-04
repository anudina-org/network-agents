FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variable declarations — values injected at runtime from OCP Secrets.
# In OCP: create a Secret and reference it via envFrom or secretKeyRef in the Deployment.
ENV LLM_PROVIDER="" \
    OLLAMA_MODEL="" \
    OLLAMA_BASE_URL="" \
    GOOGLE_API_KEY="" \
    GOOGLE_MODEL="" \
    LANGCHAIN_TRACING_V2="" \
    LANGCHAIN_API_KEY="" \
    LANGCHAIN_PROJECT=""

# CMD is set per-service in docker-compose.yml
