# AI-Q Ingestion Benchmark: B200 vs B300

This benchmark compares AI-Q 2.1.0 document ingestion using the LlamaIndex knowledge layer with:

- `AIQ_EXTRACT_IMAGES=true`
- `AIQ_EXTRACT_CHARTS=true`
- `AIQ_EXTRACT_TABLES=true`
- local VLM NIM: `nvidia/nemotron-nano-12b-v2-vl`
- embeddings, choose one:
  - local HF/PyTorch: `nvidia/llama-nemotron-embed-vl-1b-v2`
  - hosted NVIDIA endpoint: `nvidia/llama-nemotron-embed-vl-1b-v2`
- hosted LLMs from the default AI-Q config
- `generate_summary: false`

The goal is to compare AI-Q ingestion with image/chart extraction through a local VLM on each GPU. The local embedding NIM failed on B300 (`CUDA_ERROR_NO_BINARY_FOR_GPU` when loading kernels for `sm_103a`), but the same model works through Hugging Face/PyTorch with BF16 + SDPA, so embeddings can run locally through the small FastAPI service in this repo. Hosted NVIDIA embeddings remain a valid option when you want a simpler setup or want to avoid local embedding GPU contention. Hosted LLMs remain in the default config, and `generate_summary: false` keeps LLM latency out of ingestion timing.

The Knowledge API upload path is asynchronous:

- `POST /v1/collections/{collection}/documents`
- `GET /v1/documents/{job_id}/status`

NVIDIA documents these endpoints in the AI-Q 2.1.0 REST API docs:
https://docs.nvidia.com/aiq-blueprint/2.1.0/integration/rest-api.html

## Dataset

Use the same fixed input set on both hosts. Generate all three public benchmark packs before running the benchmark:

```bash
python3 generate_small_pdf_pack.py
python3 generate_medium_pdf_pack.py
python3 generate_stress_pdf_pack.py
```

The benchmark commands assume these directories exist:

```text
datasets/small-pack/pdfs/*.pdf
datasets/medium-pack/pdfs/*.pdf
datasets/stress-pack/pdfs/*.pdf
```

Suggested tiers:

- Small: 5-10 PDFs, 50-100 total pages
- Medium: 20-30 PDFs, 300-600 total pages
- Stress: 2+ large report PDFs, 1,000+ total pages

Keep each pack's `manifest.json` and `manifest.csv` with benchmark results.

To reproduce the exact B300-captured files on B200, do not re-run discovery. Copy the B300 manifests to the B200 clone, then restore each pack from those manifests:

```bash
cd ~/aiq-ingestion

python3 restore_pdf_pack_from_manifest.py aiq-ingestion-manifests/small-pack/manifest.json \
  --outdir datasets/small-pack

python3 restore_pdf_pack_from_manifest.py aiq-ingestion-manifests/medium-pack/manifest.json \
  --outdir datasets/medium-pack

python3 restore_pdf_pack_from_manifest.py aiq-ingestion-manifests/stress-pack/manifest.json \
  --outdir datasets/stress-pack
```

Before running the benchmark, verify the restored files against the manifests:

```bash
python3 restore_pdf_pack_from_manifest.py aiq-ingestion-manifests/small-pack/manifest.json \
  --outdir datasets/small-pack --verify-only

python3 restore_pdf_pack_from_manifest.py aiq-ingestion-manifests/medium-pack/manifest.json \
  --outdir datasets/medium-pack --verify-only

python3 restore_pdf_pack_from_manifest.py aiq-ingestion-manifests/stress-pack/manifest.json \
  --outdir datasets/stress-pack --verify-only
```

## Local Services

Only one local NIM should be running. If using local HF embeddings, also run the host-side embedding service:

```text
aiq-vlm-nim  localhost:8001
local HF embedding service  localhost:8010  # local_hf embedding mode only
```

Do not run `aiq-embed-nim` or `aiq-llm-nim` for the ingestion benchmark.

Warm and verify before each measured run:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/v1/health/ready

curl -X POST http://localhost:8010/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["warmup"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

For hosted embeddings, warm and verify the hosted endpoint instead:

```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":["warmup"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

Use `nvidia-smi` to confirm the expected local services are consuming benchmark GPU memory:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
nvidia-smi
```

## AI-Q Config

Use a copy of NVIDIA's default LlamaIndex config with summaries disabled:

```bash
cp configs/config_web_default_llamaindex.yml configs/config_web_ingestion_benchmark_llamaindex.yml
```

In `configs/config_web_ingestion_benchmark_llamaindex.yml`:

```yaml
functions:
  knowledge_search:
    generate_summary: false
```

In `deploy/.env`:

```bash
BACKEND_CONFIG=/app/configs/config_web_ingestion_benchmark_llamaindex.yml
LOCAL_NIM_CACHE=/localhome/local-jspaulding/.cache/nim-b300-aiq
HOST_UID=1000
HOST_GID=1000
NIM_KVCACHE_PERCENT=0.75

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
```

Use `id -u` and `id -g` for the real `HOST_UID` and `HOST_GID` values. Create `LOCAL_NIM_CACHE` as a fresh path your user owns, then verify that `$LOCAL_NIM_CACHE/vlm/local_cache` is writable before starting compose.

The hosted LLMs remain in the default `llms:` block. Since summaries are disabled, hosted LLM latency should not materially affect ingestion timing. Embedding latency is part of the benchmark, so compare medians across repeated runs and avoid mixing local and hosted embedding modes between B200 and B300.

The compose override should run the VLM NIM as your host user and let `aiq-agent` reach the host-side embedding service:

```yaml
services:
  aiq-vlm-nim:
    user: "${HOST_UID}:${HOST_GID}"
    environment:
      NIM_KVCACHE_PERCENT: ${NIM_KVCACHE_PERCENT:-0.75}
    volumes:
      - ${LOCAL_NIM_CACHE}:/opt/nim/.cache

  aiq-agent:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

For local HF embeddings, start the embedding service before ingestion:

```bash
cd ~/aiq-ingestion
source ~/venvs/nemotron-embed-vl-b300/bin/activate

python -m pip install -r requirements-local-hf-embedding-service.txt

export HF_EMBED_REVISION=0c6f636ed4c022e427277c4c336054d6cdffaa87
export HF_EMBED_ATTN=sdpa
export HF_EMBED_DTYPE=bfloat16
export CUDA_MODULE_LOADING=LAZY

python -m uvicorn local_hf_embedding_service:app --host 0.0.0.0 --port 8010 --workers 1
```

For hosted embeddings, do not start the local embedding service. Use:

```bash
AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
AIQ_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1
```

## Run The Benchmark

Run these commands from the `aiq-ingestion` repo, where `benchmark_aiq_ingestion.py` and the generated `datasets/` directory live. Leave the AI-Q Docker stack running from the separate `~/aiq` checkout.

Run B200:

```bash
cd ~/aiq-ingestion

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-ingestion-small-run1 \
  --outdir benchmark-results \
  datasets/small-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-ingestion-medium-run1 \
  --outdir benchmark-results \
  datasets/medium-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-ingestion-stress-run1 \
  --outdir benchmark-results \
  datasets/stress-pack/pdfs/*.pdf
```

Repeat each tier at least three times:

```bash
python3 benchmark_aiq_ingestion.py --label b200-ingestion-small-run2 datasets/small-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b200-ingestion-small-run3 datasets/small-pack/pdfs/*.pdf

# Repeat the same run2/run3 pattern for medium and stress.
```

Run the same commands on B300, only changing `b200` to `b300` in the label:

```bash
python3 benchmark_aiq_ingestion.py --label b300-ingestion-small-run1 datasets/small-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b300-ingestion-medium-run1 datasets/medium-pack/pdfs/*.pdf
python3 benchmark_aiq_ingestion.py --label b300-ingestion-stress-run1 datasets/stress-pack/pdfs/*.pdf
```

Each run writes:

- `benchmark-results/<label>-<timestamp>.json`
- `benchmark-results/<label>-<timestamp>-gpu.csv`

During polling, mixed status lines such as `PROCESSING,SUCCESS,INGESTING,UPLOADING` mean the asynchronous job is still running. A measured run is complete only after every reported job/document status is terminal, or as soon as any failure status appears.

The JSON includes:

- total elapsed ingestion time
- upload time
- documents per minute
- MB per minute
- final ingestion status
- input file sizes

The CSV includes `nvidia-smi` samples:

- GPU memory used
- GPU utilization
- memory utilization
- power draw
- temperature

Watch the backend logs during medium and stress runs:

```bash
docker logs -f aiq-agent | egrep -i '429|rate|limit|retry|embedding|integrate.api|8010|host.docker.internal|connection|timeout|error'
```

If the local embedding service is not reachable from Docker, or the hosted endpoint is rate-limiting, fix or label that before collecting a benchmark result.

## What To Report

For each hardware/tier, report the median of three runs:

```text
hardware, tier, summaries_enabled, files, total_mb, total_pages, total_visuals,
median_total_seconds, median_docs_per_minute, median_mb_per_minute,
embedding_mode, peak_vram_mb, mean_gpu_util_pct, peak_power_w
```

Use:

```text
summaries_enabled=false
```

Keep the raw JSON, CSV, and dataset manifests as backup.

## Interpretation

If B200 and B300 are close, ingestion is probably bottlenecked by PDF parsing, Python/LlamaIndex processing, upload overhead, ChromaDB writes, or serialized VLM calls.

If GPU utilization is low while ingest time is high, look at:

- AI-Q backend CPU usage
- PDF parsing time
- number of embedded images/charts
- whether ingestion is serializing files/images
- ChromaDB write latency

If B200 runs out of memory, stop any local embedding or local LLM containers and keep only:

```text
aiq-vlm-nim
```

## Notes

- Always use a fresh collection per run. The benchmark script does this automatically.
- Reusing an existing collection can hide work due to duplicate handling or cached state.
- Leave NIM model caches warm. You are benchmarking ingestion, not model download.
- Keep the same VLM NIM image tag across B200 and B300.
- Keep `AIQ_VLM_BASE_URL` pointed at the local Docker service name inside AI-Q, not `localhost`.
- Keep `AIQ_EMBED_BASE_URL` pointed at the same kind of endpoint on both systems, either local HF/PyTorch on both hosts or hosted on both hosts.
- Label hosted embedding runs as `embedding_mode=hosted_nvidia`.
- Label local HF/PyTorch embedding runs as `embedding_mode=local_hf`.
