# AI-Q 2.1.0 Ingestion Benchmark With Local VLM + Embeddings

This runbook deploys the AI-Q 2.1.0 full pipeline LlamaIndex example for document-ingestion benchmarking with:

- Local VLM NIM: `nvidia/nemotron-nano-12b-v2-vl`
- Local embedding NIM: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Hosted LLMs from the default AI-Q config
- `generate_summary: false` so ingestion timing does not include LLM summary generation

This is the recommended setup for B200 vs B300 ingestion benchmarking. It keeps the GPU workload focused on VLM extraction and embeddings instead of adding local LLM VRAM contention.

## References

- AI-Q 2.1.0 full pipeline LlamaIndex example: https://docs.nvidia.com/aiq-blueprint/2.1.0/examples/full-pipeline-llamaindex.html
- AI-Q 2.1.0 Docker Compose deployment: https://docs.nvidia.com/aiq-blueprint/2.1.0/deployment/docker-compose.html
- AI-Q 2.1.0 knowledge layer multimodal variables: https://docs.nvidia.com/aiq-blueprint/2.1.0/customization/knowledge-layer.html
- NeMo Retriever embedding NIM support matrix: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/2.0.0/support-matrix.html
- VLM NIM support matrix for Nemotron Nano 12B v2 VL: https://docs.nvidia.com/nim/vision-language-models/1.6.0/support-matrix.html

## 1. Host Prerequisites

On Ubuntu 22.04, install/verify:

- NVIDIA driver new enough for B200/B300 and CUDA 13-era containers.
- Docker Engine 23+ and Docker Compose v2.
- NVIDIA Container Toolkit.
- NGC/NVIDIA API key with access to NIM containers and hosted NVIDIA endpoints.

Quick GPU/container check:

```bash
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
```

## 2. Checkout AI-Q 2.1.0

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/aiq.git
cd aiq
git checkout v2.1.0
cp deploy/.env.example deploy/.env
```

## 3. Edit `deploy/.env`

Use your real keys and local cache path. On the hosted GPU machines used in this benchmark, `/localhome/local-jspaulding/.cache/nim` is the right pattern.

```bash
NVIDIA_API_KEY=nvapi-...
NGC_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...

LOCAL_NIM_CACHE=/localhome/local-jspaulding/.cache/nim
BACKEND_CONFIG=/app/configs/config_web_ingestion_benchmark_llamaindex.yml
AIQ_CHROMA_DIR=/app/data/chroma_data

AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
AIQ_EMBED_BASE_URL=http://aiq-embed-nim:8000/v1

AIQ_EXTRACT_TABLES=true
AIQ_EXTRACT_IMAGES=true
AIQ_EXTRACT_CHARTS=true
AIQ_VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
AIQ_VLM_BASE_URL=http://aiq-vlm-nim:8000/v1

NAT_JOB_STORE_DB_URL=postgresql+asyncpg://aiq:aiq_dev@postgres:5432/aiq_jobs
AIQ_CHECKPOINT_DB=postgresql://aiq:aiq_dev@postgres:5432/aiq_checkpoints
AIQ_SUMMARY_DB=postgresql+psycopg://aiq:aiq_dev@postgres:5432/aiq_jobs
```

Create the cache directory:

```bash
mkdir -p /localhome/local-jspaulding/.cache/nim
```

If you previously created the cache as another user or with root-owned files:

```bash
CACHE="$(awk -F= '/^LOCAL_NIM_CACHE=/{print $2}' deploy/.env)"
chmod -R a+rwX "$CACHE"
```

## 4. Create Ingestion Benchmark Config

Start from NVIDIA's default hosted-LLM LlamaIndex config:

```bash
cp configs/config_web_default_llamaindex.yml configs/config_web_ingestion_benchmark_llamaindex.yml
```

In `configs/config_web_ingestion_benchmark_llamaindex.yml`, set:

```yaml
functions:
  knowledge_search:
    generate_summary: false
```

Do not replace the `llms:` block with local LLM settings. Leaving the default hosted LLM block avoids local LLM VRAM contention and keeps ingestion timing focused on VLM extraction plus embeddings.

## 5. Create Compose Override

Create `deploy/compose/docker-compose.ingestion-local-nims.yaml`:

```yaml
services:
  aiq-embed-nim:
    image: nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0
    container_name: aiq-embed-nim
    gpus: all
    ipc: host
    shm_size: "16gb"
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
    volumes:
      - ${LOCAL_NIM_CACHE}:/opt/nim/.cache
    ports:
      - "8002:8000"

  aiq-vlm-nim:
    image: nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:1.6.0
    container_name: aiq-vlm-nim
    gpus: all
    ipc: host
    shm_size: "32gb"
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
      MPLCONFIGDIR: /tmp/matplotlib
    volumes:
      - ${LOCAL_NIM_CACHE}:/opt/nim/.cache
    ports:
      - "8001:8000"

  aiq-agent:
    depends_on:
      - aiq-embed-nim
      - aiq-vlm-nim
    environment:
      AIQ_EMBED_MODEL: ${AIQ_EMBED_MODEL}
      AIQ_EMBED_BASE_URL: ${AIQ_EMBED_BASE_URL}
      AIQ_EXTRACT_TABLES: ${AIQ_EXTRACT_TABLES}
      AIQ_EXTRACT_IMAGES: ${AIQ_EXTRACT_IMAGES}
      AIQ_EXTRACT_CHARTS: ${AIQ_EXTRACT_CHARTS}
      AIQ_VLM_MODEL: ${AIQ_VLM_MODEL}
      AIQ_VLM_BASE_URL: ${AIQ_VLM_BASE_URL}
```

## 6. Login and Start

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

cd deploy/compose
docker compose --env-file ../.env \
  -f docker-compose.yaml \
  -f docker-compose.ingestion-local-nims.yaml \
  up -d --build
```

First startup can take several minutes while model artifacts download and profiles initialize.

## 7. Smoke Tests

From the host:

```bash
curl http://localhost:8002/v1/health/ready
curl http://localhost:8001/v1/health/ready
curl http://localhost:8000/v1/knowledge/health
```

Embedding test:

```bash
curl -X POST http://localhost:8002/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

There is no local LLM smoke test in this benchmark setup.

## 8. Run Ingestion Benchmark

Use the generated dataset packs:

```bash
python3 benchmark_aiq_ingestion.py --label b200-small-run1 datasets/small-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-medium-run1 datasets/medium-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-stress-run1 datasets/stress-pack/pdfs/*.pdf
```

Run the same commands on B300 with labels such as `b300-small-run1`.

## 9. Operational Notes

- Re-ingest documents after changing embedding models or embedding endpoints.
- The VLM NIM is used during ingestion for image/chart extraction, not for every query.
- If AI-Q cannot reach local NIMs from inside Docker, check that base URLs use service names such as `http://aiq-embed-nim:8000/v1`, not `localhost`.
- If you keep `generate_summary: false`, hosted LLM latency should not be part of the ingestion benchmark.
- If you later benchmark full query/research workflows, label that separately because it exercises hosted LLM calls and is not a pure ingestion benchmark.
