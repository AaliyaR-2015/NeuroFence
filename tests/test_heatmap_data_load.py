"""
Mid-Project Review -- "Data Load": validate the desktop app can handle
massive JSON payloads for the heatmap, per the project's Mid-Project
Review milestone.
"""

import json
import random
import tempfile
import time
from pathlib import Path

from ui.heatmap_widget import aggregate_sweep_log


def _write_massive_sweep_log(path: Path, n_runs: int, n_layers: int) -> None:
    rng = random.Random(0)
    categories = ["benign", "edge_case", "trigger_candidate"]
    runs = []
    for i in range(n_runs):
        category = categories[i % 3]
        layer_stats = [
            {
                "layer_name": f"h.{layer_idx}",
                "mean": rng.uniform(-2, 2),
                "std": rng.uniform(0, 1),
                "max_abs": rng.uniform(0, 5),
                "shape": [1, 8, 64],
            }
            for layer_idx in range(n_layers)
        ]
        runs.append({"prompt": f"prompt {i}", "category": category, "layer_stats": layer_stats})

    with open(path, "w") as f:
        json.dump({"runs": runs}, f)


def test_aggregate_handles_50000_prompt_sweep_log_quickly():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "massive_sweep_log.json"
        _write_massive_sweep_log(path, n_runs=50_000, n_layers=12)
        file_size_mb = path.stat().st_size / (1024 * 1024)

        start = time.monotonic()
        matrix, row_labels, col_labels = aggregate_sweep_log(path)
        elapsed = time.monotonic() - start

    assert len(row_labels) == 12
    assert len(col_labels) >= 3
    assert len(matrix) == 12
    for row in matrix:
        assert len(row) == len(col_labels)

    assert elapsed < 60.0, f"aggregation took {elapsed:.2f}s for a {file_size_mb:.1f}MB file -- too slow"
    print(f"Aggregated {file_size_mb:.1f}MB / 50,000 prompts in {elapsed:.3f}s")


def test_widget_renders_massive_aggregated_matrix_offscreen():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from ui.heatmap_widget import HeatmapWidget

    app = QApplication.instance() or QApplication([])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "massive_sweep_log.json"
        _write_massive_sweep_log(path, n_runs=50_000, n_layers=12)
        matrix, row_labels, col_labels = aggregate_sweep_log(path)

    widget = HeatmapWidget()
    widget.resize(900, 500)
    widget.set_data(matrix, row_labels, col_labels)
    widget.repaint()
    assert widget._matrix == matrix


if __name__ == "__main__":
    test_aggregate_handles_50000_prompt_sweep_log_quickly()
    test_widget_renders_massive_aggregated_matrix_offscreen()
    print("All data-load stress tests passed.")
