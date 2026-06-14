#!/usr/bin/env python3
"""
Restore a PDF dataset pack from a saved manifest.

Use this when the benchmark manifests were captured on one host, such as B300,
and another host, such as B200, must ingest the exact same files. The script
downloads each manifest entry by filename, then verifies SHA-256 and byte size.

Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, user_agent: str, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    part_path = path.with_suffix(path.suffix + ".part")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        with part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    part_path.replace(path)


def download_candidates(item: dict[str, object]) -> list[str]:
    candidates: list[str] = []
    raw_candidates = item.get("download_candidates")
    if isinstance(raw_candidates, list):
        candidates.extend(str(value) for value in raw_candidates if value)

    for key in ("download_url", "url"):
        value = item.get(key)
        if value:
            candidates.append(str(value))

    return list(dict.fromkeys(candidates))


def verify_file(path: Path, item: dict[str, object]) -> tuple[bool, str]:
    expected_sha = str(item.get("sha256") or "")
    expected_bytes = item.get("bytes")

    if not path.is_file():
        return False, "missing"

    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        return False, f"size mismatch: got {path.stat().st_size}, expected {expected_bytes}"

    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            return False, f"sha256 mismatch: got {actual_sha}, expected {expected_sha}"

    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore PDFs from a benchmark manifest.")
    parser.add_argument("manifest", help="Path to manifest.json captured from the source host.")
    parser.add_argument("--outdir", required=True, help="Dataset pack directory, e.g. datasets/small-pack.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files; do not download.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout per file in seconds.")
    parser.add_argument(
        "--user-agent",
        default="aiq-ingestion-benchmark/1.0 (contact: benchmark-owner@example.com)",
        help="HTTP User-Agent. Set this to your org/contact if needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    pdf_dir = outdir / "pdfs"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SystemExit(f"{manifest_path} does not contain a files list")

    pdf_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    restored = 0
    reused = 0

    for index, item in enumerate(files, start=1):
        if not isinstance(item, dict):
            failures += 1
            print(f"[{index}/{len(files)}] invalid manifest entry", file=sys.stderr)
            continue

        filename = str(item.get("filename") or "")
        if not filename:
            failures += 1
            print(f"[{index}/{len(files)}] missing filename", file=sys.stderr)
            continue

        path = pdf_dir / filename
        ok, reason = verify_file(path, item)
        if ok and not args.force:
            reused += 1
            print(f"[{index}/{len(files)}] verified {filename}")
            continue

        if args.verify_only:
            failures += 1
            print(f"[{index}/{len(files)}] {filename}: {reason}", file=sys.stderr)
            continue

        if path.exists() and args.force:
            path.unlink()

        candidates = download_candidates(item)
        if not candidates:
            failures += 1
            print(f"[{index}/{len(files)}] {filename}: no download URL", file=sys.stderr)
            continue

        last_error: Exception | None = None
        for url in candidates:
            print(f"[{index}/{len(files)}] downloading {filename} from {url}")
            try:
                download_file(url, path, user_agent=args.user_agent, timeout=args.timeout)
                ok, reason = verify_file(path, item)
                if ok:
                    restored += 1
                    break
                raise RuntimeError(reason)
            except Exception as exc:  # noqa: BLE001 - try all manifest URLs.
                last_error = exc
                if path.exists():
                    path.unlink()
                part_path = path.with_suffix(path.suffix + ".part")
                if part_path.exists():
                    part_path.unlink()
                time.sleep(1)
        else:
            failures += 1
            print(f"[{index}/{len(files)}] {filename}: failed: {last_error}", file=sys.stderr)

    shutil.copy2(manifest_path, outdir / "manifest.json")
    csv_path = manifest_path.with_suffix(".csv")
    if csv_path.is_file():
        shutil.copy2(csv_path, outdir / "manifest.csv")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Output: {outdir}")
    print(f"Verified existing: {reused}")
    print(f"Downloaded/restored: {restored}")
    print(f"Failures: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
