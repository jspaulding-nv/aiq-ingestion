# AI-Q 2.1.0 Ingestion Benchmark With Local VLM

This runbook deploys the AI-Q 2.1.0 full pipeline LlamaIndex example for document-ingestion benchmarking with:

- Local VLM NIM: `nvidia/nemotron-nano-12b-v2-vl`
- Embeddings, choose one:
  - Local HF/PyTorch: `nvidia/llama-nemotron-embed-vl-1b-v2`
  - Hosted NVIDIA endpoint: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Hosted LLMs from the default AI-Q config
- `generate_summary: false` so ingestion timing does not include LLM summary generation

This is the recommended setup for B200 vs B300 ingestion benchmarking after the local embedding NIM failed on B300 with `CUDA_ERROR_NO_BINARY_FOR_GPU` while trying to load `sm_100a` kernels on an `sm_103a` GPU. The Hugging Face/PyTorch path works on B300 with BF16 + SDPA, so local embeddings can run as a tiny host-side FastAPI service while avoiding NIM, ONNX, TensorRT, vLLM, and flash-attn. The hosted NVIDIA endpoint remains a valid option when you want a simpler setup or want to avoid local embedding GPU contention; label those runs separately from local-HF embedding runs.

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

Use your real keys and a fresh local cache path that your user owns. Avoid reusing an existing shared or root-owned NIM cache; if the mounted cache is not writable by the NIM process, the VLM container exits with `PermissionError: /opt/nim/.cache/local_cache`.

To get the correct host-specific values:

```bash
echo "LOCAL_NIM_CACHE=$HOME/.cache/nim-b300-aiq"
echo "HOST_UID=$(id -u)"
echo "HOST_GID=$(id -g)"
```

```bash
NVIDIA_API_KEY=nvapi-...
NGC_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...

LOCAL_NIM_CACHE=/localhome/local-jspaulding/.cache/nim-b300-aiq
HOST_UID=1000
HOST_GID=1000
NIM_KVCACHE_PERCENT=0.75

BACKEND_CONFIG=/app/configs/config_web_ingestion_benchmark_llamaindex.yml
AIQ_CHROMA_DIR=/app/data/chroma_data

# Embedding mode: local_hf
AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
AIQ_EMBED_BASE_URL=http://host.docker.internal:8010/v1

# Embedding mode: hosted_nvidia
# AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
# AIQ_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1

AIQ_EXTRACT_TABLES=true
AIQ_EXTRACT_IMAGES=true
AIQ_EXTRACT_CHARTS=true
AIQ_VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
AIQ_VLM_BASE_URL=http://aiq-vlm-nim:8000/v1

NAT_JOB_STORE_DB_URL=postgresql+asyncpg://aiq:aiq_dev@postgres:5432/aiq_jobs
AIQ_CHECKPOINT_DB=postgresql://aiq:aiq_dev@postgres:5432/aiq_checkpoints
AIQ_SUMMARY_DB=postgresql+psycopg://aiq:aiq_dev@postgres:5432/aiq_jobs
```

Create and verify the local NIM cache directory before starting compose:

```bash
cd ~/aiq

CACHE="$(awk -F= '/^LOCAL_NIM_CACHE=/{print $2}' deploy/.env | tail -1)"
test -n "$CACHE" || { echo "LOCAL_NIM_CACHE is not set in deploy/.env"; exit 1; }

case "$CACHE" in
  *'$'*|~*) echo "Use an absolute expanded path for LOCAL_NIM_CACHE, not $CACHE"; exit 1 ;;
esac

mkdir -p "$CACHE/vlm/local_cache"
chmod -R ugo+rwx "$CACHE"

touch "$CACHE/vlm/local_cache/write-test" && rm "$CACHE/vlm/local_cache/write-test"
ls -ld "$CACHE" "$CACHE/vlm" "$CACHE/vlm/local_cache"
```

Do not continue until the `touch` command succeeds. If it fails, change `LOCAL_NIM_CACHE` in `deploy/.env` to a new path under your home directory, then rerun the preflight.

If you already started compose and VLM readiness failed with `/opt/nim/.cache is read-only` or `PermissionError: /opt/nim/.cache/local_cache`, stop the failed service, rerun the cache preflight above, then recreate the affected services:

```bash
cd ~/aiq/deploy/compose

docker compose --env-file ../.env \
  -f docker-compose.yaml \
  -f docker-compose.ingestion-local-vlm.yaml \
  rm -sf aiq-vlm-nim

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
    user: "${HOST_UID}:${HOST_GID}"
    ipc: host
    shm_size: "32gb"
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
      MPLCONFIGDIR: /tmp/matplotlib
      NIM_KVCACHE_PERCENT: ${NIM_KVCACHE_PERCENT:-0.75}
    volumes:
      - ${LOCAL_NIM_CACHE}:/opt/nim/.cache
    ports:
      - "8001:8000"

  aiq-agent:
    depends_on:
      - aiq-vlm-nim
    # Required for local_hf embeddings. Harmless when using hosted_nvidia.
    extra_hosts:
      - "host.docker.internal:host-gateway"
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

## 7. Choose Embedding Mode

### Option A: Local HF/PyTorch Embeddings

Run this service on the host, outside Docker, from the `aiq-ingestion` repo. It exposes an OpenAI-compatible `/v1/embeddings` endpoint and uses the exact HF revision verified on B300:

```bash
cd ~/aiq-ingestion
source ~/venvs/nemotron-embed-vl-b300/bin/activate

python -m pip install -r requirements-local-hf-embedding-service.txt

export HF_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
export HF_EMBED_REVISION=0c6f636ed4c022e427277c4c336054d6cdffaa87
export HF_EMBED_ATTN=sdpa
export HF_EMBED_DTYPE=bfloat16
export CUDA_MODULE_LOADING=LAZY

python -m uvicorn local_hf_embedding_service:app --host 0.0.0.0 --port 8010 --workers 1
```

Keep this process running while AI-Q ingests documents. The service uses a single process and an internal model lock because it is GPU-bound.

In a second shell, verify it:

```bash
cd ~/aiq-ingestion
source ~/venvs/nemotron-embed-vl-b300/bin/activate

curl http://localhost:8010/health
python3 smoke_local_embedding_service.py --base-url http://localhost:8010/v1
```

Use this AI-Q env:

```bash
AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
AIQ_EMBED_BASE_URL=http://host.docker.internal:8010/v1
```

### Option B: Hosted NVIDIA Embeddings

Do not start the local embedding service. Use this AI-Q env instead:

```bash
AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
AIQ_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1
```

Verify the hosted endpoint from the host:

```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

## 8. Smoke Tests

From the host:

```bash
curl http://localhost:8001/v1/health/ready
curl http://localhost:8000/v1/knowledge/health
```

Local HF embedding test:

```bash
curl -X POST http://localhost:8010/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

Hosted embedding test:

```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

There is no local LLM smoke test in this benchmark setup.

If readiness remains unhealthy, inspect the VLM startup logs first:

```bash
docker logs --tail 200 aiq-vlm-nim
```

## 9. Run Ingestion Benchmark

Run the benchmark from the separate `aiq-ingestion` repo, where `benchmark_aiq_ingestion.py` and the generated `datasets/` directory live. Leave the AI-Q Docker stack running in `~/aiq`.

```bash
cd ~/aiq-ingestion

python3 benchmark_aiq_ingestion.py --label b200-small-run1 datasets/small-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-medium-run1 datasets/medium-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-stress-run1 datasets/stress-pack/pdfs/*.pdf
```

Run the same commands on B300 with labels such as `b300-small-run1`.

## 10. Operational Notes

- Re-ingest documents after changing embedding models or embedding endpoints.
- The VLM NIM is used during ingestion for image/chart extraction, not for every query.
- If AI-Q cannot reach the local VLM from inside Docker, check that `AIQ_VLM_BASE_URL` uses the service name `http://aiq-vlm-nim:8000/v1`, not `localhost`.
- If AI-Q cannot reach local embeddings from inside Docker, check that `AIQ_EMBED_BASE_URL` uses `http://host.docker.internal:8010/v1` and that the `aiq-agent` service has `extra_hosts: ["host.docker.internal:host-gateway"]`.
- Embedding latency is part of the benchmark. Use the same embedding mode on B200 and B300, or label the runs separately as `local_hf` and `hosted_nvidia`.
- Hosted embeddings avoid local embedding GPU contention but can introduce API/network latency and rate limits.
- If you keep `generate_summary: false`, hosted LLM latency should not be part of the ingestion benchmark.
- If you later benchmark full query/research workflows, label that separately because it exercises hosted LLM calls and is not a pure ingestion benchmark.

Watch AI-Q backend logs for embedding errors:

```bash
docker logs -f aiq-agent | egrep -i '429|rate|limit|retry|embedding|integrate.api|8010|host.docker.internal|connection|timeout|error'
```
