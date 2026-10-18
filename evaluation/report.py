"""Turn evaluation results into evaluation/report.md, results.json and a recall chart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scoring import summarize

LABELS = {
    "detector": "Anomaly agent",
    "fixed_threshold": "Fixed 20% threshold",
    "moving_avg_z": "Moving-average z-score",
}
MIN_TARGET_MAGNITUDE = 0.2


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate raw case results into the tables used by the report."""
    det = payload["detection"]
    results = det["results"]
    first = next(iter(results.values()))
    types = list(dict.fromkeys(r.case.anomaly_type for r in first))
    magnitudes = sorted({r.case.magnitude for r in first})

    by_cell: dict[str, dict[str, Any]] = {}
    by_magnitude: dict[str, dict[str, Any]] = {}
    big: dict[str, Any] = {}
    for name, rs in results.items():
        by_cell[name] = {
            f"{t}|{m}": summarize([r for r in rs if r.case.anomaly_type == t and r.case.magnitude == m])
            for t in types
            for m in magnitudes
        }
        by_magnitude[name] = {str(m): summarize([r for r in rs if r.case.magnitude == m]) for m in magnitudes}
        big[name] = summarize([r for r in rs if r.case.magnitude >= MIN_TARGET_MAGNITUDE - 1e-9])

    attr = payload.get("attribution", {})
    attr_acc = attr.get("top1_in_dimension")
    d_big = big["detector"]
    fa = det["false_alarms_per_100d"]["detector"]
    targets = [
        ("Recall, injected anomalies of 20% or more", ">= 0.85", _num(d_big["recall"]), (d_big["recall"] or 0) >= 0.85),
        ("False alarms per 100 clean days", "<= 1", _num(fa), fa <= 1),
        (
            "Median detection delay in days (20% or more)",
            "<= 2",
            _num(d_big["median_delay"], 1),
            d_big["median_delay"] is not None and d_big["median_delay"] <= 2,
        ),
        ("Top-1 attribution accuracy", ">= 0.8", _num(attr_acc), attr_acc is not None and attr_acc >= 0.8),
    ]
    return {
        "types": types,
        "magnitudes": magnitudes,
        "by_cell": by_cell,
        "by_magnitude": by_magnitude,
        "big": big,
        "targets": targets,
        "false_alarms_per_100d": det["false_alarms_per_100d"],
        "clean_alert_counts": det["clean_alert_counts"],
    }


def render_markdown(payload: dict[str, Any], summary: dict[str, Any], chart_name: str | None) -> str:
    setup, det = payload["setup"], payload["detection"]
    names = list(summary["by_cell"])
    out = ["# Evaluation report", ""]
    out.append(
        f"Metric: **{setup['metric']}** (daily total), dataset `{setup['dataset']}`. "
        f"Replay window {det['eval_start'].date()} to {det['eval_end'].date()} ({det['n_clean_days']} days). "
        f"Each anomaly type and size was injected at {len(det['dates'])} dates "
        f"({', '.join(str(d.date()) for d in det['dates'])}). Seed {setup['seed']}."
    )
    out += [
        "",
        "Each run replays the pipeline day by day: on every date the method only sees data up to that date.",
        "",
    ]

    out += ["## Results against targets", ""]
    rows = [[t, goal, got, "met" if ok else "missed"] for t, goal, got, ok in summary["targets"]]
    out += [_table(["Target", "Goal", "Result", "Status"], rows), ""]

    out += ["## Detection by anomaly type and size (anomaly agent)", ""]
    rows = []
    for t in summary["types"]:
        for m in summary["magnitudes"]:
            s = summary["by_cell"]["detector"][f"{t}|{m}"]
            rows.append(
                [
                    t,
                    f"{m:.0%}",
                    str(s["cases"]),
                    _pct(s["recall"]),
                    _pct(s["precision"]),
                    _num(s["f1"]),
                    _num(s["median_delay"], 1),
                ]
            )
    out += [_table(["Type", "Size", "Cases", "Recall", "Precision", "F1", "Median delay (days)"], rows), ""]
    out += [
        "Recall is per injected case; precision is per alert raised that was not already present on clean data.",
        "",
    ]

    out += ["## Method comparison: recall by type and size", ""]
    rows = []
    for t in summary["types"]:
        for m in summary["magnitudes"]:
            rows.append([t, f"{m:.0%}"] + [_pct(summary["by_cell"][n][f"{t}|{m}"]["recall"]) for n in names])
    out += [_table(["Type", "Size"] + [LABELS.get(n, n) for n in names], rows), ""]
    if chart_name:
        out += [f"![Recall by anomaly size]({chart_name})", ""]

    out += ["## False alarms on clean data", ""]
    rows = [
        [LABELS.get(n, n), str(summary["clean_alert_counts"][n]), _num(summary["false_alarms_per_100d"][n])]
        for n in names
    ]
    out += [_table(["Method", "Distinct alerts", "Per 100 days"], rows), ""]
    out += [
        (
            "The clean series is the real DataCo history, so genuine anomalies in it count as false alarms. "
            "This makes the numbers conservative."
        ),
        "",
    ]

    attr = payload.get("attribution", {})
    out += ["## Attribution accuracy", ""]
    if attr.get("n"):
        out += [
            (
                f"A segment's value on one day was scaled by {attr['magnitude']:.0%} in {attr['n']} cases, "
                "and the attribution engine was asked to explain the resulting move in the total."
            ),
            "",
            _table(
                ["Measure", "Accuracy"],
                [
                    [
                        "Injected segment is the top contributor within its own dimension",
                        _pct(attr["top1_in_dimension"]),
                    ],
                    ["Injected segment is the top contributor of the headline story", _pct(attr["top1_story"])],
                ],
            ),
            "",
            (
                "The headline story picks the dimension where one segment explains the most. DataCo dimensions are "
                "partly nested (a region sits inside a market), so a parent segment can win that choice."
            ),
            "",
        ]
    else:
        out += [f"Not run: {attr.get('skipped', 'no cases')}.", ""]

    out += [
        "## Limits",
        "",
        "- Anomalies are injected by scaling aggregated daily values (attribution cases scale raw rows).",
        "- Variance increases and missing-data gaps are not covered yet.",
        "- Results come from one dataset and one seed; sample sizes per cell are small, so treat them as indicative.",
        "",
    ]
    return "\n".join(out)


def write_chart(summary: dict[str, Any], path: Path) -> bool:
    """Grouped bars of recall by anomaly size. Returns False if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = list(summary["by_magnitude"])
    mags = summary["magnitudes"]
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, n in enumerate(names):
        vals = [summary["by_magnitude"][n][str(m)]["recall"] or 0 for m in mags]
        ax.bar([x + i * width for x in range(len(mags))], vals, width, label=LABELS.get(n, n))
    ax.set_xticks([x + width * (len(names) - 1) / 2 for x in range(len(mags))])
    ax.set_xticklabels([f"{m:.0%}" for m in mags])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Injected anomaly size")
    ax.set_ylabel("Recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(payload)
    chart = out_dir / "recall_by_size.png"
    chart_name = chart.name if write_chart(summary, chart) else None
    report = out_dir / "report.md"
    report.write_text(render_markdown(payload, summary, chart_name), encoding="utf-8")
    results = out_dir / "results.json"
    results.write_text(
        json.dumps({"summary": summary, "attribution": payload.get("attribution", {})}, indent=2, default=str),
        encoding="utf-8",
    )
    return {"report": report, "results": results}
