#!/usr/bin/env python3
"""
Generate a small public PDF benchmark pack for AI-Q ingestion.

Default source: PubMed Central Open Access PDFs. The script writes:
  - PDFs under <outdir>/pdfs
  - manifest.json
  - manifest.csv
  - README.md

Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import string
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_FTP_HTTPS_PREFIX = "https://ftp.ncbi.nlm.nih.gov/"
PMC_FTP_DEPRECATED_PREFIX = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/"


def safe_filename(value: str, fallback: str) -> str:
    allowed = string.ascii_letters + string.digits + "._-"
    cleaned = "".join(ch if ch in allowed else "-" for ch in value).strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned[:140] or fallback


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    return max(1, len(re.findall(rb"/Type\s*/Page\b", data)))


def estimate_pdf_images(path: Path) -> int:
    data = path.read_bytes()
    return len(re.findall(rb"/Subtype\s*/Image\b", data))


def normalize_download_url(url: str) -> str:
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://ftp.ncbi.nlm.nih.gov/" + url.removeprefix("ftp://ftp.ncbi.nlm.nih.gov/")
    return url


def pmc_download_url_candidates(url: str) -> list[str]:
    """Return current and legacy PMC OA download locations for a discovered URL."""
    normalized = normalize_download_url(url)
    candidates = [normalized]

    # In April 2026 NCBI moved legacy PMC OA files under /pub/pmc/deprecated/.
    # The OA API can still return old /pub/pmc/oa_pdf/... URLs, so try the
    # relocated path before giving up.
    if normalized.startswith(PMC_FTP_HTTPS_PREFIX + "pub/pmc/"):
        suffix = normalized.removeprefix(PMC_FTP_HTTPS_PREFIX + "pub/pmc/")
        candidates.append(PMC_FTP_DEPRECATED_PREFIX + suffix)

    return list(dict.fromkeys(candidates))


def fetch_url(url: str, user_agent: str, timeout: int, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry any transient network/download error.
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def discover_pmc_candidates(
    start_date: str,
    until_date: str | None,
    rows_limit: int,
    user_agent: str,
    timeout: int,
) -> list[dict[str, str]]:
    params = {"from": start_date, "format": "pdf"}
    if until_date:
        params["until"] = until_date
    url = PMC_OA_URL + "?" + urllib.parse.urlencode(params)
    raw = fetch_url(url, user_agent=user_agent, timeout=timeout)
    root = ET.fromstring(raw)

    candidates: list[dict[str, str]] = []
    for record in root.findall(".//record"):
        pmcid = record.attrib.get("id", "")
        citation = record.attrib.get("citation", "")
        license_name = record.attrib.get("license", "")
        retracted = record.attrib.get("retracted", "")
        if retracted.lower() == "yes":
            continue

        for link in record.findall("link"):
            if link.attrib.get("format") != "pdf":
                continue
            href = link.attrib.get("href", "")
            if not href:
                continue
            candidates.append(
                {
                    "source": "pmc_oa",
                    "pmcid": pmcid,
                    "citation": citation,
                    "license": license_name,
                    "updated": link.attrib.get("updated", ""),
                    "url": normalize_download_url(href),
                    "download_candidates": pmc_download_url_candidates(href),
                }
            )
            break
        if len(candidates) >= rows_limit:
            break

    return candidates


def write_manifest(outdir: Path, manifest: list[dict[str, object]], settings: dict[str, object]) -> None:
    json_path = outdir / "manifest.json"
    csv_path = outdir / "manifest.csv"

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": settings,
        "total_files": len(manifest),
        "total_bytes": sum(int(item["bytes"]) for item in manifest),
        "estimated_total_pages": sum(int(item["estimated_pages"]) for item in manifest),
        "estimated_total_image_xobjects": sum(int(item["estimated_image_xobjects"]) for item in manifest),
        "files": manifest,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    fields = [
        "filename",
        "source",
        "pmcid",
        "url",
        "download_url",
        "sha256",
        "bytes",
        "estimated_pages",
        "estimated_image_xobjects",
        "license",
        "citation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in manifest:
            writer.writerow({field: item.get(field, "") for field in fields})


def write_readme(outdir: Path, manifest: list[dict[str, object]], settings: dict[str, object]) -> None:
    total_pages = sum(int(item["estimated_pages"]) for item in manifest)
    total_bytes = sum(int(item["bytes"]) for item in manifest)
    total_images = sum(int(item["estimated_image_xobjects"]) for item in manifest)
    readme = f"""# AI-Q Small PDF Benchmark Pack

Generated from PubMed Central Open Access PDF records.

- Files: {len(manifest)}
- Estimated pages: {total_pages}
- Total size: {total_bytes / 1_000_000:.2f} MB
- Estimated PDF image XObjects: {total_images}
- Source API: {PMC_OA_URL}

Use with:

```bash
python3 benchmark_aiq_ingestion.py --label b200-small-run1 {outdir / "pdfs"}/*.pdf
```

Notes:

- Page and image counts are lightweight PDF estimates, not authoritative annotations.
- Keep `manifest.json` and `manifest.csv` with benchmark results so B200 and B300 runs use the same files.
- Licenses are copied from the PMC OA API response per file.

Settings:

```json
{json.dumps(settings, indent=2, sort_keys=True)}
```
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small PDF benchmark pack for AI-Q ingestion.")
    parser.add_argument("--outdir", default="datasets/small-pack", help="Output directory.")
    parser.add_argument("--target-files", type=int, default=8, help="Target number of PDFs.")
    parser.add_argument("--min-total-pages", type=int, default=50, help="Minimum estimated pages in the pack.")
    parser.add_argument("--max-total-pages", type=int, default=100, help="Maximum estimated pages in the pack.")
    parser.add_argument("--min-pages-per-file", type=int, default=4, help="Skip PDFs below this estimated page count.")
    parser.add_argument("--max-pages-per-file", type=int, default=20, help="Skip PDFs above this estimated page count.")
    parser.add_argument("--start-date", default="2025-01-01", help="PMC OA updated-from date, YYYY-MM-DD.")
    parser.add_argument("--until-date", default="2025-03-31", help="PMC OA updated-until date, YYYY-MM-DD.")
    parser.add_argument("--max-candidates", type=int, default=250, help="Maximum PMC records to inspect.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for deterministic candidate ordering.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout per request in seconds.")
    parser.add_argument(
        "--user-agent",
        default="aiq-ingestion-benchmark/1.0 (contact: benchmark-owner@example.com)",
        help="HTTP User-Agent. Set this to your org/contact if needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pdf_dir = outdir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    settings = vars(args).copy()
    settings["outdir"] = str(outdir)

    print(f"Discovering PMC OA PDF candidates from {args.start_date} to {args.until_date}")
    candidates = discover_pmc_candidates(
        start_date=args.start_date,
        until_date=args.until_date,
        rows_limit=args.max_candidates,
        user_agent=args.user_agent,
        timeout=args.timeout,
    )
    if not candidates:
        raise SystemExit("No PMC OA PDF candidates found. Try a wider date range.")

    random.Random(args.seed).shuffle(candidates)

    manifest: list[dict[str, object]] = []
    total_pages = 0
    failures = 0

    for candidate in candidates:
        if len(manifest) >= args.target_files and total_pages >= args.min_total_pages:
            break

        pmcid = candidate["pmcid"]
        basename = safe_filename(Path(urllib.parse.urlparse(candidate["url"]).path).name, f"{pmcid}.pdf")
        filename = safe_filename(f"{pmcid}-{basename}", f"{pmcid}.pdf")
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        path = pdf_dir / filename

        try:
            if not path.exists():
                data = None
                selected_url = ""
                for url in candidate.get("download_candidates", [candidate["url"]]):
                    print(f"Downloading {pmcid}: {url}")
                    try:
                        data = fetch_url(url, user_agent=args.user_agent, timeout=args.timeout)
                        selected_url = url
                        break
                    except Exception as exc:  # noqa: BLE001 - try the next candidate URL.
                        print(f"Failed candidate {url}: {exc}")
                if data is None:
                    raise RuntimeError("all download URL candidates failed")
                if not data.startswith(b"%PDF"):
                    raise RuntimeError("download did not look like a PDF")
                path.write_bytes(data)
            else:
                print(f"Reusing existing {path.name}")
                selected_url = str(candidate["url"])

            pages = estimate_pdf_pages(path)
            if pages < args.min_pages_per_file or pages > args.max_pages_per_file:
                print(f"Skipping {path.name}: estimated pages {pages} outside per-file range")
                path.unlink(missing_ok=True)
                continue
            if total_pages + pages > args.max_total_pages:
                print(f"Skipping {path.name}: would exceed max total pages")
                path.unlink(missing_ok=True)
                continue

            item = {
                **candidate,
                "filename": path.name,
                "path": str(path),
                "download_url": selected_url,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "estimated_pages": pages,
                "estimated_image_xobjects": estimate_pdf_images(path),
            }
            manifest.append(item)
            total_pages += pages
            print(f"Selected {path.name}: pages={pages}, total_pages={total_pages}")

        except Exception as exc:  # noqa: BLE001 - continue through bad records.
            failures += 1
            print(f"Failed {pmcid}: {exc}")

    write_manifest(outdir, manifest, settings)
    write_readme(outdir, manifest, settings)

    print()
    print(f"Wrote pack to {outdir}")
    print(f"Selected files: {len(manifest)}")
    print(f"Estimated pages: {total_pages}")
    print(f"Download/parse failures skipped: {failures}")
    print(f"Manifest: {outdir / 'manifest.json'}")

    if len(manifest) < args.target_files or total_pages < args.min_total_pages:
        print()
        print("Warning: target not fully met. Try increasing --max-candidates or widening --until-date.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
