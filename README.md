# AI-Q Ingestion Benchmark

This repo contains helper scripts and runbooks for benchmarking NVIDIA AI-Q 2.1.0 document ingestion on B200/B300 systems.

The benchmark focuses on AI-Q's LlamaIndex knowledge layer with:

- local VLM extraction through `nvidia/nemotron-nano-12b-v2-vl`
- embeddings through either local Hugging Face/PyTorch or the hosted NVIDIA endpoint
- hosted LLMs from the default AI-Q config
- `generate_summary: false` so ingestion timing does not include summary-generation latency

## Start Here

Use the B300 runbook for setup and operations:

- [AI-Q 2.1.0 B300 local VLM runbook](aiq-2.1.0-b300-local-nims-runbook.md)

Use the benchmark guide for methodology, dataset handling, run commands, and reporting:

- [B200 vs B300 ingestion benchmark](aiq-ingestion-b200-vs-b300-benchmark.md)

## Repositories

This helper repo is used alongside the NVIDIA AI-Q repo:

```bash
cd ~
git clone https://github.com/NVIDIA-AI-Blueprints/aiq.git
cd ~/aiq
git checkout v2.1.0

cd ~
git clone <your-aiq-ingestion-repo-url> aiq-ingestion
```

Run AI-Q from `~/aiq`. Run dataset generation, local embedding service, and benchmark commands from `~/aiq-ingestion`.

## Embedding Modes

The local embedding NIM path failed on B300 because the NIM/runtime stack tried to use incompatible GPU kernels for the B300 target. This repo documents two supported alternatives:

```text
embedding_mode=local_hf
AIQ_EMBED_BASE_URL=http://host.docker.internal:8010/v1
```

```text
embedding_mode=hosted_nvidia
AIQ_EMBED_BASE_URL=https://integrate.api.nvidia.com/v1
```

Keep the same embedding mode across B200 and B300 when comparing hardware, or label results separately.

## Local HF Embedding Service

For `embedding_mode=local_hf`, start the host-side service from this repo:

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

Smoke test it:

```bash
python3 smoke_local_embedding_service.py --base-url http://localhost:8010/v1
```

## Dataset Helpers

Generate benchmark PDF packs:

```bash
python3 generate_small_pdf_pack.py
python3 generate_medium_pdf_pack.py
python3 generate_stress_pdf_pack.py
```

Restore fixed packs from manifests:

```bash
python3 restore_pdf_pack_from_manifest.py path/to/manifest.json --outdir datasets/small-pack
```

## Benchmark Runner

Run ingestion benchmarks against a running AI-Q backend:

```bash
python3 benchmark_aiq_ingestion.py \
  --base-url http://localhost:8000 \
  --label b300-ingestion-small-run1 \
  --outdir benchmark-results \
  datasets/small-pack/pdfs/*.pdf
```

Each run writes a JSON result and an `nvidia-smi` GPU sample CSV under `benchmark-results/`.
