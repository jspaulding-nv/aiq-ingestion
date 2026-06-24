#!/usr/bin/env python3
"""Smoke test the local HF embedding service with standard-library HTTP."""

from __future__ import annotations

import argparse
import json
import urllib.request


def post_json(url: str, body: object, timeout: int = 120) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a local embedding service.")
    parser.add_argument("--base-url", default="http://localhost:8010/v1")
    parser.add_argument("--model", default="nvidia/llama-nemotron-embed-vl-1b-v2")
    args = parser.parse_args()

    response = post_json(
        f"{args.base_url.rstrip('/')}/embeddings",
        {
            "model": args.model,
            "input": ["hello from local embeddings", "Q2 revenue increased 42 percent"],
            "input_type": "passage",
            "modality": "text",
        },
    )
    data = response["data"]
    dimensions = len(data[0]["embedding"])
    print(f"embeddings={len(data)} dimensions={dimensions} model={response.get('model')}")
    if dimensions != 2048:
        raise SystemExit(f"unexpected embedding dimension: {dimensions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

