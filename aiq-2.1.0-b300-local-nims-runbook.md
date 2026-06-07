# AI-Q 2.1.0 Full Pipeline With Local NIMs on One B300

This runbook deploys the AI-Q 2.1.0 full pipeline LlamaIndex example with these local NIMs on a single B300 GPU:

- LLM: `nvidia/nemotron-3-nano-30b-a3b` through a local Nemotron 3 Nano NIM
- Embeddings: `nvidia/llama-nemotron-embed-vl-1b-v2`
- VLM extraction: `nvidia/nemotron-nano-12b-v2-vl`

Web search is still external if you keep the Tavily tools enabled in the default AI-Q example.

## References

- AI-Q 2.1.0 full pipeline LlamaIndex example: https://docs.nvidia.com/aiq-blueprint/2.1.0/examples/full-pipeline-llamaindex.html
- AI-Q 2.1.0 Docker Compose deployment: https://docs.nvidia.com/aiq-blueprint/2.1.0/deployment/docker-compose.html
- AI-Q 2.1.0 knowledge layer multimodal variables: https://docs.nvidia.com/aiq-blueprint/2.1.0/customization/knowledge-layer.html
- Nemotron 3 Nano LLM NIM support matrix: https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html
- LLM NIM GPU memory behavior: https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html
- NeMo Retriever embedding NIM getting started: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/1.12.0/getting-started.html
- VLM NIM support matrix for Nemotron Nano 12B v2 VL: https://docs.nvidia.com/nim/vision-language-models/1.6.0/support-matrix.html

## 1. Host Prerequisites

On Ubuntu 22.04, install/verify:

- NVIDIA driver new enough for B300 and CUDA 13-era containers. NVIDIA VLM NIM 1.7 docs recommend driver 580+.
- Docker Engine 23+ and Docker Compose v2.
- NVIDIA Container Toolkit.
- NGC/NVIDIA API key with access to NIM containers.
- Tavily API key if you keep web search enabled.

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

Use your real keys and paths:

```bash
NVIDIA_API_KEY=nvapi-...
NGC_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...

LOCAL_NIM_CACHE=/home/ubuntu/.cache/nim
BACKEND_CONFIG=/app/configs/config_web_local_llamaindex.yml
AIQ_CHROMA_DIR=/app/data/chroma_data

AIQ_LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b
AIQ_LLM_BASE_URL=http://aiq-llm-nim:8000/v1

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
mkdir -p /home/ubuntu/.cache/nim
```

## 4. Create Local AI-Q Config

```bash
cp configs/config_web_default_llamaindex.yml configs/config_web_local_llamaindex.yml
```

In `configs/config_web_local_llamaindex.yml`, replace the entire `llms:` block with this local-only version. Keep the existing `functions:` and `workflow:` sections unchanged.

```yaml
llms:
  nemotron_llm_intent:
    _type: nim
    model_name: ${AIQ_LLM_MODEL:-nvidia/nemotron-3-nano-30b-a3b}
    base_url: ${AIQ_LLM_BASE_URL:-http://aiq-llm-nim:8000/v1}
    temperature: 0.5
    top_p: 0.9
    max_tokens: 4096
    num_retries: 3
    chat_template_kwargs:
      enable_thinking: true

  nemotron_nano_llm:
    _type: nim
    model_name: ${AIQ_LLM_MODEL:-nvidia/nemotron-3-nano-30b-a3b}
    base_url: ${AIQ_LLM_BASE_URL:-http://aiq-llm-nim:8000/v1}
    temperature: 0.1
    top_p: 0.3
    max_tokens: 16384
    num_retries: 3
    chat_template_kwargs:
      enable_thinking: true

  summary_llm:
    _type: nim
    model_name: ${AIQ_LLM_MODEL:-nvidia/nemotron-3-nano-30b-a3b}
    base_url: ${AIQ_LLM_BASE_URL:-http://aiq-llm-nim:8000/v1}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 150
    num_retries: 3
    chat_template_kwargs:
      enable_thinking: false
```

If the local LLM NIM's `/v1/models` response returns `nvidia/nemotron-3-nano` instead of `nvidia/nemotron-3-nano-30b-a3b`, set:

```bash
AIQ_LLM_MODEL=nvidia/nemotron-3-nano
```

## 5. Create Compose Override

Create `deploy/compose/docker-compose.local-nims.yaml`:

```yaml
services:
  aiq-llm-nim:
    image: nvcr.io/nim/nvidia/nemotron-3-nano:2.0.5
    container_name: aiq-llm-nim
    gpus: all
    ipc: host
    shm_size: "16gb"
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
      NIM_MODEL_PROFILE: vllm-nvfp4-tp1-pp1
      NIM_LOG_LEVEL: INFO
      PYTORCH_ALLOC_CONF: expandable_segments:True
    volumes:
      - ${LOCAL_NIM_CACHE}:/opt/nim/.cache
    ports:
      - "8003:8000"
    command:
      - --gpu-memory-utilization
      - "0.45"
      - --max-model-len
      - "32768"

  aiq-embed-nim:
    image: nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0
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
    volumes:
      - ${LOCAL_NIM_CACHE}:/opt/nim/.cache
    ports:
      - "8001:8000"

  aiq-agent:
    depends_on:
      - aiq-llm-nim
      - aiq-embed-nim
      - aiq-vlm-nim
    environment:
      AIQ_LLM_MODEL: ${AIQ_LLM_MODEL}
      AIQ_LLM_BASE_URL: ${AIQ_LLM_BASE_URL}
      AIQ_EMBED_MODEL: ${AIQ_EMBED_MODEL}
      AIQ_EMBED_BASE_URL: ${AIQ_EMBED_BASE_URL}
      AIQ_EXTRACT_TABLES: ${AIQ_EXTRACT_TABLES}
      AIQ_EXTRACT_IMAGES: ${AIQ_EXTRACT_IMAGES}
      AIQ_EXTRACT_CHARTS: ${AIQ_EXTRACT_CHARTS}
      AIQ_VLM_MODEL: ${AIQ_VLM_MODEL}
      AIQ_VLM_BASE_URL: ${AIQ_VLM_BASE_URL}
```

The LLM NIM is capped at 45% of B300 VRAM because vLLM-backed NIMs greedily allocate remaining memory for KV cache. If you need longer context, increase `--max-model-len` and/or `--gpu-memory-utilization`, then watch `nvidia-smi`.

## 6. Login and Start

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

cd deploy/compose
docker compose --env-file ../.env \
  -f docker-compose.yaml \
  -f docker-compose.local-nims.yaml \
  up -d --build
```

First startup can take several minutes while model artifacts download and profiles initialize.

## 7. Smoke Tests

From the host:

```bash
curl http://localhost:8003/v1/models
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

LLM test:

```bash
curl -X POST http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/nemotron-3-nano-30b-a3b","messages":[{"role":"user","content":"Say ready."}],"max_tokens":32}'
```

If this LLM test returns a model-not-found error, use the model ID returned by `/v1/models` and update `AIQ_LLM_MODEL`.

## 8. Use the Pipeline

Open the UI:

```text
http://localhost:3000
```

Or use the API:

```bash
curl -X POST http://localhost:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"my-docs","description":"Local LlamaIndex documents"}'

curl -X POST http://localhost:8000/v1/collections/my-docs/documents \
  -F "files=@report.pdf"

curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"shallow_researcher","input":"What does the uploaded report say about GPU memory?"}'
```

Poll or stream the returned `job_id`:

```bash
curl -N http://localhost:8000/v1/jobs/async/job/<job_id>/stream
```

## 9. Operational Notes

- Re-ingest documents after changing embedding models or embedding endpoints.
- The VLM NIM is used during ingestion for image/chart extraction, not for every query.
- If AI-Q cannot reach local NIMs from inside Docker, check that base URLs use service names such as `http://aiq-llm-nim:8000/v1`, not `localhost`.
- If the LLM NIM OOMs during graph capture or warmup, lower `--gpu-memory-utilization` to `0.40` or set `NIM_DISABLE_CUDA_GRAPH=1`.
- If you do not want external web search, remove `web_search_tool` and `advanced_web_search_tool` from the YAML and from each agent's `tools:` list.
