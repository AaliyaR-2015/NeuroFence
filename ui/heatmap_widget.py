"""
heatmap_widget.py
-------------------
Week 2 scope, per the project plan: "Build a visual matrix in the UI (like
a heatmap) representing the active vs. dormant neurons in the model."

Two pieces:
    - aggregate_sweep_log(): reads the JSON artifact produced by
      src/fuzz_runner.py (Member 2) and reduces it to a small
      layers x categories matrix of mean activation. This only depends on
      the JSON *shape*, not on importing fuzz_runner's Python code -- a
      deliberate seam so the UI branch doesn't need the AI-forensics
      branch's code merged in just to compile/test.
    - HeatmapWidget: a plain QWidget that paints that matrix as a colored
      grid using QPainter directly (no charting library dependency,
      consistent with this project's "no web components, minimal
      dependencies" native-app philosophy).
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

# Category order is fixed (rather than "whatever order first appears in the
# log") so the heatmap's columns are always in the same, predictable
# left-to-right order no matter which sweep log is loaded.
CATEGORY_ORDER = ["benign", "edge_case", "trigger_candidate"]


def aggregate_sweep_log(path: str | Path) -> tuple[list[list[float]], list[str], list[str]]:
    """
    Read a fuzz_runner sweep log JSON file and reduce it to:

        matrix[row][col] = mean activation of `row_labels[row]` layer,
                            averaged over every prompt in
                            `col_labels[col]` category.

    Returns (matrix, row_labels, col_labels). Layers are ordered by their
    first appearance in the log (which, for the hooked-block naming used
    by ActivationTracker, is also their depth order in the network).
    """
    data = json.loads(Path(path).read_text())
    runs = data["runs"]

    # sums[layer][category] = (total, count) for incremental averaging.
    sums: dict[str, dict[str, list[float]]] = {}
    layer_order: list[str] = []

    for run in runs:
        category = run["category"]
        for layer_stat in run["layer_stats"]:
            layer_name = layer_stat["layer_name"]
            if layer_name not in sums:
                sums[layer_name] = {cat: [0.0, 0] for cat in CATEGORY_ORDER}
                layer_order.append(layer_name)
            if category not in sums[layer_name]:
                # Defensive: an unexpected category in the log shouldn't crash
                # the UI, just gets its own column bucket.
                sums[layer_name][category] = [0.0, 0]
            total, count = sums[layer_name][category]
            sums[layer_name][category] = [total + layer_stat["mean"], count + 1]

    all_categories = list(CATEGORY_ORDER)
    for layer_sums in sums.values():
        for cat in layer_sums:
            if cat not in all_categories:
                all_categories.append(cat)

    matrix = []
    for layer_name in layer_order:
        row = []
        for cat in all_categories:
            total, count = sums[layer_name].get(cat, [0.0, 0])
            row.append(total / count if count else 0.0)
        matrix.append(row)

    return matrix, layer_order, all_categories


def _value_to_color(t: float) -> QColor:
    """
    Map a normalized value t in [0, 1] to a blue -> yellow -> red heat
    color. Blue = dormant/low activation, red = highly active -- the same
    convention the project doc uses ("active vs. dormant neurons").
    """
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        # blue -> yellow
        local_t = t / 0.5
        r = int(40 + local_t * (240 - 40))
        g = int(60 + local_t * (220 - 60))
        b = int(200 + local_t * (60 - 200))
    else:
        # yellow -> red
        local_t = (t - 0.5) / 0.5
        r = int(240 + local_t * (220 - 240))
        g = int(220 + local_t * (40 - 220))
        b = int(60 + local_t * (40 - 60))
    return QColor(r, g, b)


class HeatmapWidget(QWidget):
    """Paints a layers x categories matrix as a colored grid."""

    LABEL_MARGIN_LEFT = 90
    LABEL_MARGIN_TOP = 24
    MIN_CELL_SIZE = 28

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._matrix: list[list[float]] = []
        self._row_labels: list[str] = []
        self._col_labels: list[str] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, matrix: list[list[float]], row_labels: list[str], col_labels: list[str]) -> None:
        self._matrix = matrix
        self._row_labels = row_labels
        self._col_labels = col_labels
        self.updateGeometry()
        self.update()  # schedule a repaint

    def sizeHint(self):
        from PyQt5.QtCore import QSize

        n_rows = max(1, len(self._matrix))
        n_cols = max(1, len(self._col_labels))
        width = self.LABEL_MARGIN_LEFT + n_cols * self.MIN_CELL_SIZE + 20
        height = self.LABEL_MARGIN_TOP + n_rows * self.MIN_CELL_SIZE + 20
        return QSize(width, height)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._matrix or not self._col_labels:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No fuzz sweep log loaded yet.")
            return

        n_rows = len(self._matrix)
        n_cols = len(self._col_labels)

        flat_values = [v for row in self._matrix for v in row]
        lo, hi = min(flat_values), max(flat_values)
        value_range = (hi - lo) or 1.0  # avoid division by zero if all values equal

        available_w = max(1, self.width() - self.LABEL_MARGIN_LEFT - 10)
        available_h = max(1, self.height() - self.LABEL_MARGIN_TOP - 10)
        cell_w = max(self.MIN_CELL_SIZE, available_w / n_cols)
        cell_h = max(self.MIN_CELL_SIZE, available_h / n_rows)

        # Column headers (category names).
        painter.setPen(QColor("#222222"))
        for col, label in enumerate(self._col_labels):
            x = self.LABEL_MARGIN_LEFT + col * cell_w
            rect = QRectF(x, 0, cell_w, self.LABEL_MARGIN_TOP)
            painter.drawText(rect, Qt.AlignCenter, label)

        # Rows: label + colored cells.
        for row_idx, (row, layer_name) in enumerate(zip(self._matrix, self._row_labels)):
            y = self.LABEL_MARGIN_TOP + row_idx * cell_h
            label_rect = QRectF(0, y, self.LABEL_MARGIN_LEFT - 6, cell_h)
            painter.setPen(QColor("#222222"))
            painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignRight, layer_name)

            for col_idx, value in enumerate(row):
                x = self.LABEL_MARGIN_LEFT + col_idx * cell_w
                normalized = (value - lo) / value_range
                color = _value_to_color(normalized)

                cell_rect = QRectF(x, y, cell_w, cell_h)
                painter.fillRect(cell_rect, color)
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawRect(cell_rect)
