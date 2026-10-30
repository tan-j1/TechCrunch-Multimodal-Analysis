"""Aggregate timestamped video evidence into measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.metrics import summarize


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Saved per-video evidence JSON")
    p.add_argument("--output", type=Path, required=True, help="Summary CSV")
    p.add_argument("--config", type=Path, default=root / "configs/multimodal.yaml")
    a = p.parse_args()
    row = summarize(json.loads(a.input.read_text(encoding="utf-8")), load_config(a.config))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
