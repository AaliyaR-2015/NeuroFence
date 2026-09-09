"""
anomaly_report_widget.py — Week 3

New, standalone widget (doesn't touch heatmap_widget.py or main_window.py --
see integration_notes.md for the ~10-line wiring diff to add a second
"Load Anomaly Report..." button next to your existing one).

Renders outputs/anomaly_report.json (from src/anomaly_detector.py) as a
sorted table: most suspicious ("selective trigger") findings first, plain
statistical outliers below, colored red/orange so a reviewer can tell "the
scanner found a likely backdoor" at a glance without reading raw JSON.
"""

import json

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt


COLUMNS = ["Layer", "Neuron", "Selective Trigger?", "Benign z", "Trigger z", "Baseline mean", "Baseline std"]


class AnomalyReportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.summary_label = QLabel("No report loaded.")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

    def load_report(self, path: str):
        with open(path) as f:
            report = json.load(f)

        findings = report.get("findings", [])
        num_selective = report.get("num_selective_trigger_findings", 0)

        if num_selective > 0:
            self.summary_label.setText(
                f"⚠ {num_selective} likely backdoor trigger neuron(s) found "
                f"out of {len(findings)} total anomalies flagged."
            )
        else:
            self.summary_label.setText(f"No selective-trigger neurons found ({len(findings)} other anomalies).")

        self.table.setRowCount(len(findings))
        for row, finding in enumerate(findings):
            values = [
                finding["layer"],
                str(finding["neuron_idx"]),
                "YES" if finding["is_selective_trigger"] else "no",
                f"{finding['benign_z']:.2f}",
                f"{finding['trigger_z']:.2f}",
                f"{finding['baseline_mean']:.4f}",
                f"{finding['baseline_std']:.4f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if finding["is_selective_trigger"]:
                    item.setBackground(QColor(255, 120, 120))
                elif abs(finding["trigger_z"]) > 4:
                    item.setBackground(QColor(255, 210, 140))
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
