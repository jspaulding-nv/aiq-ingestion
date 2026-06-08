# AI-Q Ingestion Benchmark: B200 vs B300

This benchmark compares AI-Q 2.1.0 document ingestion using the LlamaIndex knowledge layer with:

- `AIQ_EXTRACT_IMAGES=true`
- `AIQ_EXTRACT_CHARTS=true`
- `AIQ_EXTRACT_TABLES=true`
- local VLM NIM: `nvidia/nemotron-nano-12b-v2-vl`
- hosted embeddings: `nvidia/llama-nemotron-embed-vl-1b-v2`
- hosted LLMs from the default AI-Q config
- `generate_summary: false`

The goal is to compare AI-Q ingestion with image/chart extraction through a local VLM on each GPU. Embeddings and LLMs are hosted so the benchmark avoids local LLM VRAM contention and the current local embedding NIM failure observed on B300 (`CUDA_ERROR_NO_BINARY_FOR_GPU` when loading kernels for `sm_103a`).

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

## Local NIMs

Only one local NIM should be running:

```text
aiq-vlm-nim  localhost:8001
```

Do not run `aiq-embed-nim` or `aiq-llm-nim` for the ingestion benchmark.

Warm and verify before each measured run:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/v1/health/ready

curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":["warmup"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

Use `nvidia-smi` to confirm only the local VLM is consuming benchmark GPU memory:

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

AIQ_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
AIQ_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1

AIQ_EXTRACT_TABLES=true
AIQ_EXTRACT_IMAGES=true
AIQ_EXTRACT_CHARTS=true
AIQ_VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
AIQ_VLM_BASE_URL=http://aiq-vlm-nim:8000/v1
```

The hosted LLMs remain in the default `llms:` block. Since summaries are disabled, hosted LLM latency should not materially affect ingestion timing. Hosted embedding latency is part of the benchmark, so compare medians across repeated runs and avoid mixing local and hosted embedding modes between B200 and B300.

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
docker logs -f aiq-agent | egrep -i '429|rate|limit|retry|embedding|integrate.api'
```

If the hosted embedding endpoint throttles requests, label that run as hosted-embedding/API-limited. The small pack is the safest tier for avoiding the public API's 40 RPM limit.

## What To Report

For each hardware/tier, report the median of three runs:

```text
hardware, tier, summaries_enabled, files, total_mb, total_pages, total_visuals,
median_total_seconds, median_docs_per_minute, median_mb_per_minute,
peak_vram_mb, mean_gpu_util_pct, peak_power_w
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
- Keep `AIQ_EMBED_BASE_URL` pointed at the same hosted endpoint on both systems.
- Mark runs as hosted-embedding/API-limited if NVIDIA public endpoint rate limits affect ingestion.
