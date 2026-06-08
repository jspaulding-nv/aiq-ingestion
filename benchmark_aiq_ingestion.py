#!/usr/bin/env python3
"""
Benchmark AI-Q document ingestion through the Knowledge API.

This script intentionally uses only Python's standard library so it can run on
the target Ubuntu hosts without installing extra packages.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import random
import shutil
import string
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


TERMINAL_STATUSES = {
    "COMPLETE",
    "COMPLETED",
    "DONE",
    "ERROR",
    "FAILED",
    "FAILURE",
    "FINISHED",
    "SUCCESS",
    "SUCCEEDED",
}

FAILURE_STATUSES = {"ERROR", "FAILED", "FAILURE"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_name(value: str) -> str:
    allowed = string.ascii_letters + string.digits + "-_"
    return "".join(ch if ch in allowed else "-" for ch in value).strip("-")


def request_json(method: str, url: str, body: object | None = None, timeout: int = 60) -> object:
    headers = {}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def upload_files(base_url: str, collection: str, files: list[Path], timeout: int) -> object:
    boundary = "----aiq-benchmark-" + "".join(random.choice(string.ascii_letters) for _ in range(16))
    chunks: list[bytes] = []

    for path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                'Content-Disposition: form-data; name="files"; '
                f'filename="{path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    data = b"".join(chunks)

    encoded_collection = urllib.parse.quote(collection, safe="")
    url = f"{base_url.rstrip('/')}/v1/collections/{encoded_collection}/documents"
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc


def collect_status_values(value: object) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"status", "state", "job_status", "ingestion_status"} and isinstance(item, str):
                values.append(item)
            values.extend(collect_status_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(collect_status_values(item))
    return values


def terminal_status(status_doc: object) -> tuple[bool, bool, str]:
    values = [status.upper() for status in collect_status_values(status_doc)]
    if not values:
        return False, False, "UNKNOWN"

    failed = any(status in FAILURE_STATUSES for status in values)
    done = all(status in TERMINAL_STATUSES for status in values)
    return done, failed, ",".join(values)


class GpuSampler:
    def __init__(self, output_path: Path, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.enabled = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval_seconds + 5)

    def _run(self) -> None:
        query = (
            "--query-gpu=index,name,memory.used,memory.total,"
            "utilization.gpu,utilization.memory,power.draw,temperature.gpu"
        )
        command = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
        header = (
            "sample_time_utc,index,name,memory_used_mb,memory_total_mb,"
            "gpu_util_pct,mem_util_pct,power_w,temp_c\n"
        )
        with self.output_path.open("w", encoding="utf-8") as handle:
            handle.write(header)
            while not self.stop_event.is_set():
                sample_time = utc_now()
                result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        handle.write(f"{sample_time},{line}\n")
                    handle.flush()
                self.stop_event.wait(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AI-Q Knowledge API document ingestion.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="AI-Q backend base URL.")
    parser.add_argument("--label", required=True, help="Run label, for example b200-normalized-run1.")
    parser.add_argument("--collection", help="Collection name. Defaults to a unique name from --label.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Status polling interval in seconds.")
    parser.add_argument("--timeout", type=int, default=7200, help="Overall ingestion timeout in seconds.")
    parser.add_argument("--upload-timeout", type=int, default=900, help="Upload request timeout in seconds.")
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0, help="nvidia-smi sample interval.")
    parser.add_argument("--outdir", default="benchmark-results", help="Directory for result JSON and GPU CSV.")
    parser.add_argument("--cleanup", action="store_true", help="Delete the collection after the run.")
    parser.add_argument("files", nargs="+", help="Files to upload and ingest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    label = safe_name(args.label)
    collection = args.collection or f"bench-{label}-{int(time.time())}"
    files = [Path(file).expanduser().resolve() for file in args.files]

    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input files: {', '.join(missing)}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = int(time.time())
    result_path = outdir / f"{label}-{started}.json"
    gpu_path = outdir / f"{label}-{started}-gpu.csv"

    run_result: dict[str, object] = {
        "label": label,
        "base_url": base_url,
        "collection": collection,
        "started_at_utc": utc_now(),
        "files": [{"path": str(path), "name": path.name, "bytes": path.stat().st_size} for path in files],
        "total_bytes": sum(path.stat().st_size for path in files),
        "gpu_samples_csv": str(gpu_path),
    }

    sampler = GpuSampler(gpu_path, args.gpu_sample_interval)

    try:
        print(f"[{utc_now()}] Checking AI-Q health at {base_url}/health")
        run_result["health"] = request_json("GET", f"{base_url}/health", timeout=30)

        print(f"[{utc_now()}] Creating collection {collection}")
        run_result["create_collection_response"] = request_json(
            "POST",
            f"{base_url}/v1/collections",
            {"name": collection, "description": f"AI-Q ingestion benchmark {label}", "metadata": {"label": label}},
            timeout=60,
        )

        sampler.start()
        upload_start = time.perf_counter()
        print(f"[{utc_now()}] Uploading {len(files)} file(s)")
        upload_response = upload_files(base_url, collection, files, timeout=args.upload_timeout)
        upload_elapsed = time.perf_counter() - upload_start
        run_result["upload_response"] = upload_response
        run_result["upload_elapsed_seconds"] = upload_elapsed

        job_id = upload_response.get("job_id") if isinstance(upload_response, dict) else None
        if not job_id:
            raise RuntimeError(f"Upload response did not include job_id: {upload_response}")

        poll_start = time.perf_counter()
        deadline = time.monotonic() + args.timeout
        status_url = f"{base_url}/v1/documents/{urllib.parse.quote(str(job_id), safe='')}/status"
        status_history: list[dict[str, object]] = []
        final_status: object = {}

        print(f"[{utc_now()}] Polling ingestion job {job_id}")
        while time.monotonic() < deadline:
            status_doc = request_json("GET", status_url, timeout=60)
            done, failed, status_text = terminal_status(status_doc)
            status_history.append(
                {
                    "elapsed_seconds": round(time.perf_counter() - poll_start, 3),
                    "status_text": status_text,
                    "raw": status_doc,
                }
            )
            print(f"[{utc_now()}] status={status_text}")
            final_status = status_doc
            if done or failed:
                run_result["failed"] = failed
                break
            time.sleep(args.poll_interval)
        else:
            run_result["failed"] = True
            run_result["timeout"] = True

        ingest_elapsed = time.perf_counter() - poll_start
        total_elapsed = time.perf_counter() - upload_start
        run_result["job_id"] = job_id
        run_result["status_history"] = status_history
        run_result["final_status"] = final_status
        run_result["ingest_poll_elapsed_seconds"] = ingest_elapsed
        run_result["total_elapsed_seconds"] = total_elapsed
        run_result["ended_at_utc"] = utc_now()
        run_result["docs_per_minute"] = len(files) / total_elapsed * 60 if total_elapsed else None
        run_result["mb_per_minute"] = (run_result["total_bytes"] / 1_000_000) / total_elapsed * 60 if total_elapsed else None

    finally:
        sampler.stop()
        if args.cleanup:
            try:
                print(f"[{utc_now()}] Deleting collection {collection}")
                request_json(
                    "DELETE",
                    f"{base_url}/v1/collections/{urllib.parse.quote(collection, safe='')}",
                    timeout=120,
                )
                run_result["cleanup"] = "deleted"
            except Exception as exc:  # noqa: BLE001 - cleanup should not hide benchmark results.
                run_result["cleanup"] = f"failed: {exc}"
        run_result["gpu_sampler_enabled"] = sampler.enabled
        result_path.write_text(json.dumps(run_result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[{utc_now()}] Wrote {result_path}")
        if sampler.enabled:
            print(f"[{utc_now()}] Wrote {gpu_path}")

    return 1 if run_result.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
