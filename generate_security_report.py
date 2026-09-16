"""
generate_security_report.py — Week 4

Turns outputs/anomaly_report.json into a readable Markdown security report
-- the kind of artifact you'd actually hand to someone after a scan.
Stdlib only (json, pathlib, datetime) on purpose: no new pip installs
needed under deadline pressure. Markdown renders natively on GitHub too,
so it doubles as nice-looking evidence straight in your repo.

Usage:
    python -m src.generate_security_report --report outputs/anomaly_report.json \
        --out outputs/security_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_report_markdown(report: dict) -> str:
    meta = report.get("meta", {})
    findings = report.get("findings", [])
    selective = [f for f in findings if f.get("is_selective_trigger")]
    other = [f for f in findings if not f.get("is_selective_trigger")]

    lines = []
    lines.append("# NeuroFence Security Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Model scanned: `{meta.get('model', 'unknown')}`")
    lines.append(f"Baseline: `{meta.get('baseline', 'unknown')}`")
    lines.append(f"Z-score threshold: {meta.get('z_threshold', 'n/a')}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if selective:
        lines.append(
            f"**\u26a0\ufe0f LIKELY BACKDOOR DETECTED** \u2014 {len(selective)} neuron(s) show the "
            f"signature of a planted trigger: dormant on benign input, sharply anomalous "
            f"specifically on trigger-pattern input."
        )
    else:
        lines.append(
            "**\u2705 No selective-trigger neurons found.** No neuron in this scan was both "
            "dormant on benign input and anomalous on trigger-pattern input."
        )
    lines.append("")
    lines.append(f"- Total anomalous neurons flagged: {len(findings)}")
    lines.append(f"- Of which, selective-trigger (likely backdoor) neurons: {len(selective)}")
    lines.append("")

    if selective:
        lines.append("## Likely Backdoor Neurons")
        lines.append("")
        lines.append("| Layer | Neuron | Benign z | Trigger z | Baseline mean | Baseline std |")
        lines.append("|---|---|---|---|---|---|")
        for f in selective:
            lines.append(
                f"| {f['layer']} | {f['neuron_idx']} | {f['benign_z']:.2f} | "
                f"{f['trigger_z']:.2f} | {f['baseline_mean']:.4f} | {f['baseline_std']:.4f} |"
            )
        lines.append("")

    if other:
        lines.append("## Other Statistical Anomalies (not selective triggers)")
        lines.append("")
        lines.append("These neurons deviate from baseline but are anomalous on benign input too "
                      "-- more likely to be generally noisy than a planted backdoor.")
        lines.append("")
        lines.append("| Layer | Neuron | Benign z | Trigger z |")
        lines.append("|---|---|---|---|")
        for f in other[:20]:  # cap the table so it stays readable
            lines.append(f"| {f['layer']} | {f['neuron_idx']} | {f['benign_z']:.2f} | {f['trigger_z']:.2f} |")
        if len(other) > 20:
            lines.append(f"| ... | *{len(other) - 20} more, omitted for brevity* | | |")
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "Per-neuron activations were collected across benign, edge-case, and "
        "trigger-candidate prompt categories (see `adversarial_fuzzer.py`), then compared "
        "against a benign-only baseline (`baseline_profile.py`, Welford's algorithm) using "
        "z-scores. A neuron is flagged as a **selective trigger** when it stays within "
        "normal range on benign input but crosses the anomaly threshold specifically on "
        "trigger-pattern input -- the statistical fingerprint of a planted backdoor rather "
        "than ordinary model noise."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown security report from an anomaly_report.json")
    parser.add_argument("--report", default="outputs/anomaly_report.json")
    parser.add_argument("--out", default="outputs/security_report.md")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    markdown = build_report_markdown(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
