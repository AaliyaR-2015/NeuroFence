"""
Tests for heatmap_widget.py: both the pure JSON-aggregation logic
(no Qt needed) and the widget itself (needs QApplication, run offscreen).
"""

import json
import tempfile
from pathlib import Path

from ui.heatmap_widget import aggregate_sweep_log


def _write_sample_log(path: Path) -> None:
    runs = [
        {
            "prompt": "benign one", "category": "benign",
            "layer_stats": [
                {"layer_name": "h.0", "mean": 0.10, "std": 0.02, "max_abs": 0.5, "shape": [1, 6, 16]},
                {"layer_name": "h.1", "mean": 0.20, "std": 0.03, "max_abs": 0.6, "shape": [1, 6, 16]},
            ],
        },
        {
            "prompt": "benign two", "category": "benign",
            "layer_stats": [
                {"layer_name": "h.0", "mean": 0.30, "std": 0.02, "max_abs": 0.5, "shape": [1, 6, 16]},
                {"layer_name": "h.1", "mean": 0.40, "std": 0.03, "max_abs": 0.6, "shape": [1, 6, 16]},
            ],
        },
        {
            "prompt": "trigger candidate", "category": "trigger_candidate",
            "layer_stats": [
                {"layer_name": "h.0", "mean": 9.0, "std": 1.0, "max_abs": 12.0, "shape": [1, 6, 16]},
                {"layer_name": "h.1", "mean": 8.0, "std": 1.0, "max_abs": 11.0, "shape": [1, 6, 16]},
            ],
        },
    ]
    path.write_text(json.dumps({"runs": runs}))


def test_aggregate_averages_within_category():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sweep_log.json"
        _write_sample_log(path)

        matrix, row_labels, col_labels = aggregate_sweep_log(path)

    assert row_labels == ["h.0", "h.1"]
    assert col_labels[:3] == ["benign", "edge_case", "trigger_candidate"]

    benign_col = col_labels.index("benign")
    trigger_col = col_labels.index("trigger_candidate")
    edge_col = col_labels.index("edge_case")

    # h.0: benign mean of [0.10, 0.30] = 0.20; trigger_candidate = 9.0
    assert abs(matrix[0][benign_col] - 0.20) < 1e-9
    assert abs(matrix[0][trigger_col] - 9.0) < 1e-9
    assert matrix[0][edge_col] == 0.0  # no edge_case prompts in this sample log

    # h.1: benign mean of [0.20, 0.40] = 0.30; trigger_candidate = 8.0
    assert abs(matrix[1][benign_col] - 0.30) < 1e-9
    assert abs(matrix[1][trigger_col] - 8.0) < 1e-9


def test_layer_order_matches_first_appearance():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sweep_log.json"
        # Deliberately reference h.1 before h.0 in the first run.
        runs = [
            {
                "prompt": "x", "category": "benign",
                "layer_stats": [
                    {"layer_name": "h.1", "mean": 1.0, "std": 0.1, "max_abs": 2.0, "shape": [1, 1, 4]},
                    {"layer_name": "h.0", "mean": 1.0, "std": 0.1, "max_abs": 2.0, "shape": [1, 1, 4]},
                ],
            }
        ]
        path.write_text(json.dumps({"runs": runs}))

        _, row_labels, _ = aggregate_sweep_log(path)

    assert row_labels == ["h.1", "h.0"]


def test_widget_renders_without_crashing_offscreen():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from ui.heatmap_widget import HeatmapWidget

    app = QApplication.instance() or QApplication([])

    widget = HeatmapWidget()
    widget.resize(400, 200)
    widget.set_data(
        matrix=[[0.1, 0.9, 5.0], [0.2, 0.3, 4.5]],
        row_labels=["h.0", "h.1"],
        col_labels=["benign", "edge_case", "trigger_candidate"],
    )
    widget.repaint()  # forces paintEvent synchronously
    assert widget._matrix == [[0.1, 0.9, 5.0], [0.2, 0.3, 4.5]]


def test_widget_shows_placeholder_with_no_data():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from ui.heatmap_widget import HeatmapWidget

    app = QApplication.instance() or QApplication([])
    widget = HeatmapWidget()
    widget.resize(300, 100)
    widget.repaint()  # should not raise even with empty data


if __name__ == "__main__":
    test_aggregate_averages_within_category()
    test_layer_order_matches_first_appearance()
    test_widget_renders_without_crashing_offscreen()
    test_widget_shows_placeholder_with_no_data()
    print("All heatmap_widget tests passed.")
