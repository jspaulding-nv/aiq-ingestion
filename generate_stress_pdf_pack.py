#!/usr/bin/env python3
"""
Generate a stress PDF benchmark pack for AI-Q ingestion.

The stress pack intentionally uses large, chart-heavy public reports. It writes:
  - PDFs under <outdir>/pdfs
  - manifest.json
  - manifest.csv
  - README.md

Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import urllib.request

from generate_small_pdf_pack import (
    estimate_pdf_images,
    estimate_pdf_pages,
    safe_filename,
    sha256_file,
)


DEFAULT_REPORTS = [
    {
        "source": "ipcc",
        "title": "IPCC AR6 Synthesis Report Full Volume",
        "license": "IPCC public report; check source terms",
        "url": "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_FullVolume.pdf",
    },
    {
        "source": "ipcc",
        "title": "IPCC AR6 Working Group I Full Report",
        "license": "IPCC public report; check source terms",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FullReport.pdf",
    },
    {
        "source": "ipcc",
        "title": "IPCC AR6 Working Group II Full Report",
        "license": "IPCC public report; check source terms",
        "url": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_FullReport.pdf",
    },
    {
        "source": "ipcc",
        "title": "IPCC AR6 Working Group III Full Report",
        "license": "IPCC public report; check source terms",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_FullReport.pdf",
    },
]


def download_pdf_streaming(
    url: str,
    path: Path,
    user_agent: str,
    timeout: int,
    progress_interval: float,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    part_path = path.with_suffix(path.suffix + ".part")
    downloaded = 0
    last_progress = 0.0

    with urllib.request.urlopen(request, timeout=timeout) as response:
        total_header = response.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else None
        total_mb = total_bytes / 1_000_000 if total_bytes else None

        with part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_progress >= progress_interval:
                    downloaded_mb = downloaded / 1_000_000
                    if total_mb:
                        print(f"  downloaded {downloaded_mb:.1f}/{total_mb:.1f} MB")
                    else:
                        print(f"  downloaded {downloaded_mb:.1f} MB")
                    last_progress = now

    with part_path.open("rb") as handle:
        if handle.read(4) != b"%PDF":
            raise RuntimeError("download did not look like a PDF")

    part_path.replace(path)


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
        "title",
        "url",
        "sha256",
        "bytes",
        "estimated_pages",
        "estimated_image_xobjects",
        "license",
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
    readme = f"""# AI-Q Stress PDF Benchmark Pack

Generated from large public report PDFs.

- Files: {len(manifest)}
- Estimated pages: {total_pages}
- Total size: {total_bytes / 1_000_000:.2f} MB
- Estimated PDF image XObjects: {total_images}

Use with:

```bash
python3 benchmark_aiq_ingestion.py --label b200-stress-run1 {outdir / "pdfs"}/*.pdf
```

Notes:

- This pack is intentionally heavy. Start with the small and medium packs before running it.
- Page and image counts are lightweight PDF estimates, not authoritative annotations.
- Keep `manifest.json` and `manifest.csv` with benchmark results so B200 and B300 runs use the same files.
- The included reports are public PDFs, but always check source terms before redistributing the pack.

Settings:

```json
{json.dumps(settings, indent=2, sort_keys=True)}
```
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a stress PDF benchmark pack for AI-Q ingestion.")
    parser.add_argument("--outdir", default="datasets/stress-pack", help="Output directory.")
    parser.add_argument("--min-total-pages", type=int, default=1000, help="Minimum estimated pages in the pack.")
    parser.add_argument("--max-files", type=int, default=2, help="Maximum number of large reports to download.")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout per request in seconds.")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between large-download progress messages.",
    )
    parser.add_argument(
        "--user-agent",
        default="aiq-ingestion-benchmark/1.0 (contact: benchmark-owner@example.com)",
        help="HTTP User-Agent. Set this to your org/contact if needed.",
    )
    parser.add_argument(
        "--include-all-ipcc",
        action="store_true",
        help="Download all configured IPCC reports, ignoring --max-files and --min-total-pages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    pdf_dir = outdir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    settings = vars(args).copy()
    settings["outdir"] = str(outdir)

    manifest: list[dict[str, object]] = []
    total_pages = 0
    failures = 0

    for report in DEFAULT_REPORTS:
        if not args.include_all_ipcc and len(manifest) >= args.max_files:
            break
        if not args.include_all_ipcc and total_pages >= args.min_total_pages:
            break

        url = str(report["url"])
        filename = safe_filename(Path(url).name, "report.pdf")
        path = pdf_dir / filename

        try:
            if not path.exists():
                print(f"Downloading {report['title']}: {url}")
                download_pdf_streaming(
                    url=url,
                    path=path,
                    user_agent=args.user_agent,
                    timeout=args.timeout,
                    progress_interval=args.progress_interval,
                )
            else:
                print(f"Reusing existing {path.name}")

            pages = estimate_pdf_pages(path)
            images = estimate_pdf_images(path)
            item = {
                **report,
                "filename": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "estimated_pages": pages,
                "estimated_image_xobjects": images,
            }
            manifest.append(item)
            total_pages += pages
            print(f"Selected {path.name}: pages={pages}, images={images}, total_pages={total_pages}")

        except Exception as exc:  # noqa: BLE001 - continue through bad records.
            failures += 1
            print(f"Failed {report['title']}: {exc}")

    write_manifest(outdir, manifest, settings)
    write_readme(outdir, manifest, settings)

    print()
    print(f"Wrote pack to {outdir}")
    print(f"Selected files: {len(manifest)}")
    print(f"Estimated pages: {total_pages}")
    print(f"Download/parse failures skipped: {failures}")
    print(f"Manifest: {outdir / 'manifest.json'}")

    if total_pages < args.min_total_pages and not args.include_all_ipcc:
        print()
        print("Warning: target not fully met. Try --max-files 4 or --include-all-ipcc.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
