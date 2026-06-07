#!/usr/bin/env python3
"""
Generate a medium public PDF benchmark pack for AI-Q ingestion.

This is a thin wrapper around generate_small_pdf_pack.py with defaults sized
for roughly 20-30 files and 300-600 estimated pages.
"""

from __future__ import annotations

import sys

from generate_small_pdf_pack import main


DEFAULT_ARGS = [
    "--outdir",
    "datasets/medium-pack",
    "--target-files",
    "25",
    "--min-total-pages",
    "300",
    "--max-total-pages",
    "600",
    "--min-pages-per-file",
    "6",
    "--max-pages-per-file",
    "35",
    "--start-date",
    "2024-01-01",
    "--until-date",
    "2025-12-31",
    "--max-candidates",
    "1000",
    "--seed",
    "84",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    raise SystemExit(main())
