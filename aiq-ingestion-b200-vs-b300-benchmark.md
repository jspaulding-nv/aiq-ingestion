# AI-Q Ingestion Benchmark: B200 vs B300

This benchmark compares AI-Q 2.1.0 document ingestion using the LlamaIndex knowledge layer with:

- `AIQ_EXTRACT_IMAGES=true`
- `AIQ_EXTRACT_CHARTS=true`
- local VLM NIM: `nvidia/nemotron-nano-12b-v2-vl`
- local embedding NIM: `nvidia/llama-nemotron-embed-vl-1b-v2`
- optional local LLM NIM if document summaries remain enabled

The Knowledge API upload path is asynchronous:

- `POST /v1/collections/{collection}/documents`
- `GET /v1/documents/{job_id}/status`

NVIDIA documents these endpoints in the AI-Q 2.1.0 REST API docs:
https://docs.nvidia.com/aiq-blueprint/2.1.0/integration/rest-api.html

## Recommendation

Run two benchmark modes:

1. Normalized mode
   - Same NIM tags, same AI-Q version, same documents, same memory caps, same context limits.
   - Best answer to: "How much faster is B300 than B200 under the same deployment constraints?"

2. Best-effort mode
   - Tune B200 and B300 independently to the largest stable concurrency/KV cache.
   - Best answer to: "What is the practical throughput I can get from each GPU?"

For procurement or architecture comparison, report both. Normalized mode is the clean comparison; best-effort mode is the operational answer.

## Dataset

Use the same fixed input set on both hosts. Prefer a visual-heavy set because this benchmark is specifically about VLM ingestion.

Suggested tiers:

- Small: 5 PDFs, 50-100 total pages
- Medium: 20 PDFs, 300-600 total pages
- Stress: 2+ large report PDFs, 1,000+ total pages

Generate the public benchmark packs:

```bash
python3 generate_small_pdf_pack.py
python3 generate_medium_pdf_pack.py
python3 generate_stress_pdf_pack.py
```

The small and medium generators use PubMed Central Open Access PDF records. The stress generator uses large public IPCC report PDFs by default.

Record for each tier:

- file count
- total bytes
- approximate pages
- approximate embedded images/charts

Do not benchmark model download time. Start only after all NIMs are healthy and warmed.

## B200 One-GPU Settings

A single B200 has about 180 GB of HBM. It should run the full local setup, but the containers need explicit memory limits because vLLM-backed NIMs allocate KV cache greedily by default.

For B200, start with:

```yaml
services:
  aiq-llm-nim:
    command:
      - --gpu-memory-utilization
      - "0.12"
      - --max-model-len
      - "32768"
      - --max-num-seqs
      - "128"
    environment:
      PYTORCH_ALLOC_CONF: expandable_segments:True

  aiq-vlm-nim:
    environment:
      NIM_KVCACHE_PERCENT: "0.25"
      NIM_MAX_MODEL_LEN: "8192"
      NIM_MAX_NUM_SEQS: "1"

  aiq-embed-nim:
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
```

If you disable document summaries for an ingestion-only benchmark, the LLM NIM will not materially affect ingestion. You can still leave it running to match the full pipeline, or stop it to isolate VLM + embedding performance.

## B300 Normalized Settings

For normalized comparison, use the same conservative caps as B200 even though B300 has more memory:

```yaml
services:
  aiq-llm-nim:
    command:
      - --gpu-memory-utilization
      - "0.12"
      - --max-model-len
      - "32768"
      - --max-num-seqs
      - "128"

  aiq-vlm-nim:
    environment:
      NIM_KVCACHE_PERCENT: "0.25"
      NIM_MAX_MODEL_LEN: "8192"
      NIM_MAX_NUM_SEQS: "1"
```

For best-effort B300, raise these gradually:

```yaml
services:
  aiq-llm-nim:
    command:
      - --gpu-memory-utilization
      - "0.18"
      - --max-model-len
      - "32768"
      - --max-num-seqs
      - "256"

  aiq-vlm-nim:
    environment:
      NIM_KVCACHE_PERCENT: "0.35"
      NIM_MAX_MODEL_LEN: "16384"
      NIM_MAX_NUM_SEQS: "2"
```

## Isolate VLM Ingestion

The full LlamaIndex config may generate document summaries. If `generate_summary: true`, ingestion includes calls to `summary_llm`, which means your benchmark is VLM + embedding + LLM summary generation.

For a pure document ingestion/VLM benchmark, set this in `configs/config_web_local_llamaindex.yml`:

```yaml
functions:
  knowledge_search:
    generate_summary: false
```

For a full-pipeline benchmark, leave summaries enabled and clearly label the results as "with summaries".

## Warmup

Run this before each measured run:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/v1/health/ready
curl http://localhost:8002/v1/health/ready
curl http://localhost:8003/v1/models

curl -X POST http://localhost:8002/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["warmup"],"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","modality":"text"}'
```

If summaries are enabled, warm the LLM too:

```bash
curl -X POST http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/nemotron-3-nano-30b-a3b","messages":[{"role":"user","content":"Say ready."}],"max_tokens":8}'
```

## Run the Benchmark

After generating the three dataset tiers, benchmark each tier from this repo directory. The commands below assume these directories exist:

```text
datasets/small-pack/pdfs/*.pdf
datasets/medium-pack/pdfs/*.pdf
datasets/stress-pack/pdfs/*.pdf
```

Run B200 normalized mode:

```bash
python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-normalized-small-run1 \
  --outdir benchmark-results \
  datasets/small-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-normalized-medium-run1 \
  --outdir benchmark-results \
  datasets/medium-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-normalized-stress-run1 \
  --outdir benchmark-results \
  datasets/stress-pack/pdfs/*.pdf
```

Repeat each tier at least three times:

```bash
python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-normalized-small-run2 \
  --outdir benchmark-results \
  datasets/small-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b200-normalized-small-run3 \
  --outdir benchmark-results \
  datasets/small-pack/pdfs/*.pdf

# Repeat the same run2/run3 pattern for medium and stress.
```

Run the same commands on the B300 host, only changing `b200` to `b300` in the label:

```bash
python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b300-normalized-small-run1 \
  --outdir benchmark-results \
  datasets/small-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b300-normalized-medium-run1 \
  --outdir benchmark-results \
  datasets/medium-pack/pdfs/*.pdf

python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b300-normalized-stress-run1 \
  --outdir benchmark-results \
  datasets/stress-pack/pdfs/*.pdf
```

For best-effort mode, use the same dataset paths and change the label, for example `b200-best-effort-medium-run1` or `b300-best-effort-medium-run1`.

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

## What to Report

For each hardware/mode/tier, report median of three runs:

```text
hardware, mode, tier, summaries_enabled, files, total_mb, total_pages, total_visuals,
median_total_seconds, median_docs_per_minute, median_mb_per_minute,
peak_vram_mb, mean_gpu_util_pct, peak_power_w
```

Use median, not best run. Keep the raw JSON and CSV files as backup.

## Interpretation

If B200 and B300 are close, ingestion is probably bottlenecked by PDF parsing, Python/LlamaIndex processing, upload overhead, or serialized VLM calls.

If B300 pulls ahead mostly in best-effort mode, the difference is likely larger usable KV cache/concurrency rather than pure single-request latency.

If GPU utilization is low while ingest time is high, look at:

- AI-Q backend CPU usage
- PDF parsing time
- number of embedded images/charts
- whether ingestion is serializing files/images
- ChromaDB write latency

If B200 OOMs, lower:

```yaml
NIM_KVCACHE_PERCENT: "0.20"
NIM_MAX_NUM_SEQS: "1"
```

and lower the LLM cap:

```yaml
--gpu-memory-utilization 0.10
--max-num-seqs 128
```

## Notes

- Always use a fresh collection per run. The benchmark script does this automatically.
- Reusing an existing collection can hide work due to duplicate handling or cached state.
- Leave NIM model caches warm. You are benchmarking ingestion, not model download.
- Keep the same NIM image tags across B200 and B300.
- Keep `AIQ_EMBED_BASE_URL` and `AIQ_VLM_BASE_URL` pointed at local Docker service names inside AI-Q, not `localhost`.
