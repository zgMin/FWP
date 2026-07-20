#!/usr/bin/env python3
"""Merge Auto-Margin columns into existing aligned benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD = "qtraj_teacher_auto_margin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["verifiable", "descriptive"], required=True)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--auto", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    base_payload = json.loads(args.base.read_text(encoding="utf-8"))
    base_rows = {row["pair_id"]: row for row in base_payload["results"]}
    auto_rows = []
    for path in args.auto:
        auto_rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    if len({row["pair_id"] for row in auto_rows}) != len(auto_rows):
        raise ValueError("duplicate pair_id in Auto-Margin inputs")

    merged = []
    missing = []
    for auto in auto_rows:
        pair_id = auto["pair_id"]
        if pair_id not in base_rows:
            missing.append(pair_id)
            continue
        row = base_rows[pair_id]
        row["outputs"][METHOD] = auto["outputs"][METHOD]
        if "output_token_counts" in auto:
            row.setdefault("output_token_counts", {})[METHOD] = auto["output_token_counts"][METHOD]
        if "terminated_with_eos" in auto:
            row.setdefault("terminated_with_eos", {})[METHOD] = auto["terminated_with_eos"][METHOD]
        if args.task == "descriptive":
            row.setdefault("trajectory_kl_mean", {})[METHOD] = auto["trajectory_kl_mean"][METHOD]
            row.setdefault("token_kl", {})[METHOD] = auto["token_kl"][METHOD]
        row["auto_margin"] = auto["auto_margin"]
        row.setdefault("timing_seconds", {})["auto_margin"] = auto["timing_seconds"]["auto_margin"]
        merged.append(row)
    if missing:
        raise ValueError(f"Auto-Margin rows missing from base report: {missing[:3]}")
    merged.sort(key=lambda row: row["pair_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_config = base_payload.get("config", {})
    if not base_config and base_payload.get("configs"):
        base_config = base_payload["configs"][0]
    args.out.write_text(
        json.dumps(
            {
                "config": base_config,
                "methods": [
                    "base_no_prompt",
                    "full_prompt",
                    "qtraj_teacher_margin",
                    "qtraj_topk_delta",
                    METHOD,
                ],
                "result_count": len(merged),
                "results": merged,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "rows": len(merged)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
