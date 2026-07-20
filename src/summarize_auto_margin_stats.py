#!/usr/bin/env python3
"""Summarize Auto-Margin convergence, adaptive rank, parameters, and runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHOD = "qtraj_teacher_auto_margin"


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def distribution(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def aggregate(rows: list[dict]) -> dict:
    active = [float(row["auto_margin"]["active_constraints"]) for row in rows]
    constraints = [float(row["auto_margin"]["constraint_count"]) for row in rows]
    parameters = [float(row["auto_margin"]["anchor_parameter_count"]) for row in rows]
    solve_rounds = [float(max(0, len(row["auto_margin"]["rounds"]) - 1)) for row in rows]
    auto_seconds = [float(row["timing_seconds"]["auto_margin"]) for row in rows]
    qtraj_seconds = [float(row["timing_seconds"]["qtraj_fit"]) for row in rows]
    return {
        "rows": len(rows),
        "exact_output_match": sum(row["outputs"][METHOD] == row["outputs"]["full_prompt"] for row in rows)
        / max(len(rows), 1),
        "convergence_rate": sum(bool(row["auto_margin"]["converged"]) for row in rows) / max(len(rows), 1),
        "zero_anchor_rate": sum(value == 0 for value in active) / max(len(active), 1),
        "active_rank": distribution(active),
        "constraint_count": distribution(constraints),
        "anchor_parameter_count": distribution(parameters),
        "solve_rounds": distribution(solve_rounds),
        "auto_margin_seconds": distribution(auto_seconds),
        "qtraj_fit_seconds": distribution(qtraj_seconds),
    }


def grouped(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: aggregate(value) for key, value in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--group-fields", nargs="+", default=["language"])
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pair_id across inputs")
    report = {
        "overall": aggregate(rows),
        "groups": {field: grouped(rows, field) for field in args.group_fields},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = [
        "breakdown",
        "group",
        "rows",
        "exact_output_match",
        "convergence_rate",
        "zero_anchor_rate",
        "active_rank_mean",
        "active_rank_median",
        "active_rank_p90",
        "active_rank_p95",
        "active_rank_max",
        "anchor_parameters_mean",
        "anchor_parameters_p95",
        "anchor_parameters_max",
        "auto_margin_seconds_mean",
        "auto_margin_seconds_p95",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        breakdowns = {"overall": {"all": report["overall"]}, **report["groups"]}
        for breakdown, groups in breakdowns.items():
            for group, item in groups.items():
                writer.writerow(
                    {
                        "breakdown": breakdown,
                        "group": group,
                        "rows": item["rows"],
                        "exact_output_match": item["exact_output_match"],
                        "convergence_rate": item["convergence_rate"],
                        "zero_anchor_rate": item["zero_anchor_rate"],
                        **{f"active_rank_{key}": value for key, value in item["active_rank"].items()},
                        "anchor_parameters_mean": item["anchor_parameter_count"]["mean"],
                        "anchor_parameters_p95": item["anchor_parameter_count"]["p95"],
                        "anchor_parameters_max": item["anchor_parameter_count"]["max"],
                        "auto_margin_seconds_mean": item["auto_margin_seconds"]["mean"],
                        "auto_margin_seconds_p95": item["auto_margin_seconds"]["p95"],
                    }
                )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
