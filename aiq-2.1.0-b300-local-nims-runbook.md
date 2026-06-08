# AI-Q 2.1.0 Ingestion Benchmark With Local VLM

This runbook deploys the AI-Q 2.1.0 full pipeline LlamaIndex example for document-ingestion benchmarking with:

- Local VLM NIM: `nvidia/nemotron-nano-12b-v2-vl`
- Hosted embeddings: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Hosted LLMs from the default AI-Q config
- `generate_summary: false` so ingestion timing does not include LLM summary generation

This is the recommended setup for B200 vs B300 ingestion benchmarking after the local embedding NIM failed on B300 with `CUDA_ERROR_NO_BINARY_FOR_GPU` while trying to load `sm_100a` kernels on an `sm_103a` GPU. It keeps the local GPU workload focused on VLM extraction and avoids both local LLM VRAM contention and the current B300 embedding-kernel mismatch.

## References

- AI-Q 2.1.0 full pipeline LlamaIndex example: https://docs.nvidia.com/aiq-blueprint/2.1.0/examples/full-pipeline-llamaindex.html
- AI-Q 2.1.0 Docker Compose deployment: https://docs.nvidia.com/aiq-blueprint/2.1.0/deployment/docker-compose.html
- AI-Q 2.1.0 knowledge layer multimodal variables: https://docs.nvidia.com/aiq-blueprint/2.1.0/customization/knowledge-layer.html
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
AIQ_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1

AIQ_EXTRACT_TABLES=true
AIQ_EXTRACT_IMAGES=true
AIQ_EXTRACT_CHARTS=true
AIQ_VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
AIQ_VLM_BASE_URL=http://aiq-vlm-nim:8000/v1

NAT_JOB_STORE_DB_URL=postgresql+asyncpg://aiq:aiq_dev@postgres:5432/aiq_jobs
AIQ_CHECKPOINT_DB=postgresql://aiq:aiq_dev@postgres:5432/aiq_checkpoints
AIQ_SUMMARY_DB=postgresql+psycopg://aiq:aiq_dev@postgres:5432/aiq_jobs
```

Create the local NIM cache directory for the VLM:

```bash
sudo mkdir -p /localhome/local-jspaulding/.cache/nim/vlm
sudo chmod -R 777 /localhome/local-jspaulding/.cache/nim
```

If VLM readiness fails with `/opt/nim/.cache is read-only` or `PermissionError: /opt/nim/.cache/local_cache`, repair the mounted cache and recreate the affected services:

```bash
cd ~/aiq

CACHE="$(awk -F= '/^LOCAL_NIM_CACHE=/{print $2}' deploy/.env | tail -1)"

sudo mkdir -p "$CACHE/vlm"
sudo chmod -R 777 "$CACHE"

cd ~/aiq/deploy/compose

docker rm -f aiq-vlm-nim 2>/dev/null || true

docker compose --env-file ../.env \
  -f docker-compose.yaml \
  -f docker-compose.ingestion-local-vlm.yaml \
  up -d --force-recreate aiq-vlm-nim aiq-agent
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

Do not replace the `llms:` block with local LLM settings. Leaving the default hosted LLM block avoids local LLM VRAM contention. With `generate_summary: false`, hosted LLM latency should not materially affect ingestion timing.

## 5. Create Compose Override

Create `deploy/compose/docker-compose.ingestion-local-vlm.yaml`:

```yaml
services:
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
  -f docker-compose.ingestion-local-vlm.yaml \
  up -d --build
```

If old local embedding or local LLM containers are still present from earlier testing, stop them:

```bash
docker stop aiq-embed-nim aiq-llm-nim 2>/dev/null || true
```

First startup can take several minutes while model artifacts download and profiles initialize.

## 7. Smoke Tests

From the host:

```bash
curl http://localhost:8001/v1/health/ready
curl http://localhost:8000/v1/knowledge/health
```

Hosted embedding test:

```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

There is no local embedding or local LLM smoke test in this benchmark setup.

If readiness remains unhealthy, inspect the VLM startup logs first:

```bash
docker logs --tail 200 aiq-vlm-nim
```

## 8. Run Ingestion Benchmark

Run the benchmark from the separate `aiq-ingestion` repo, where `benchmark_aiq_ingestion.py` and the generated `datasets/` directory live. Leave the AI-Q Docker stack running in `~/aiq`.

```bash
cd ~/aiq-ingestion

python3 benchmark_aiq_ingestion.py --label b200-small-run1 datasets/small-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-medium-run1 datasets/medium-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-stress-run1 datasets/stress-pack/pdfs/*.pdf
```

Run the same commands on B300 with labels such as `b300-small-run1`.

## 9. Operational Notes

- Re-ingest documents after changing embedding models or embedding endpoints.
- The VLM NIM is used during ingestion for image/chart extraction, not for every query.
- If AI-Q cannot reach the local VLM from inside Docker, check that `AIQ_VLM_BASE_URL` uses the service name `http://aiq-vlm-nim:8000/v1`, not `localhost`.
- Hosted embedding latency is now part of the benchmark. Use repeated runs and compare medians so transient network/API variance does not dominate the result.
- The public NVIDIA hosted endpoint can be limited to 40 RPM. If medium or stress runs hit rate limits, label those results as hosted-embedding/API-limited.
- If you keep `generate_summary: false`, hosted LLM latency should not be part of the ingestion benchmark.
- If you later benchmark full query/research workflows, label that separately because it exercises hosted LLM calls and is not a pure ingestion benchmark.

Watch AI-Q backend logs for hosted embedding throttling:

```bash
docker logs -f aiq-agent | egrep -i '429|rate|limit|retry|embedding|integrate.api'
```
