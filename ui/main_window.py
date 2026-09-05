"""
main_window.py
--------------
Native, offline desktop UI skeleton for NeuroFence.

Week 1 scope (per project plan): "Initialize a local desktop application.
Create views for uploading model files and displaying basic metadata."

This module deliberately has NO network calls and no dependency on a running
model at import time -- the window can be opened and clicked around even
before a model is loaded, which is what "initialize the desktop application"
means for Week 1. Wiring it to the real model_sandbox loader happens in
main.py so this file stays testable in isolation.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "NeuroFence -- LLM Backdoor Forensic Scanner"


class MainWindow(QMainWindow):
    """
    Top-level window with:
      - a "load model" row (local path browser + HF model-id field)
      - a metadata panel (populated once a model is loaded)
      - a status/log panel (for sandbox-loading and future fuzzing output)

    `on_load_model_requested` is an injectable callback so this UI can be
    unit-tested / demoed without importing PyTorch or transformers at all --
    the real backend call is wired in from main.py.
    """

    def __init__(
        self,
        on_load_model_requested: Optional[Callable[[str], dict]] = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._on_load_model_requested = on_load_model_requested
        self.setWindowTitle(APP_TITLE)
        self.resize(760, 520)
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        root_layout.addWidget(self._build_load_model_group())
        root_layout.addWidget(self._build_metadata_group())
        root_layout.addWidget(self._build_log_group())

    def _build_load_model_group(self) -> QGroupBox:
        group = QGroupBox("Load Model", self)
        layout = QHBoxLayout(group)

        self.model_path_field = QLineEdit(group)
        self.model_path_field.setPlaceholderText(
            "HuggingFace model id (e.g. sshleifer/tiny-gpt2) or local folder path"
        )

        browse_btn = QPushButton("Browse Local Folder...", group)
        browse_btn.clicked.connect(self._on_browse_clicked)

        load_btn = QPushButton("Load into Sandbox", group)
        load_btn.setDefault(True)
        load_btn.clicked.connect(self._on_load_clicked)

        layout.addWidget(self.model_path_field, stretch=1)
        layout.addWidget(browse_btn)
        layout.addWidget(load_btn)
        return group

    def _build_metadata_group(self) -> QGroupBox:
        group = QGroupBox("Model Metadata", self)
        layout = QFormLayout(group)

        self.metadata_labels: dict[str, QLabel] = {}
        for field in [
            "name_or_path", "architecture", "num_parameters",
            "num_layers", "hidden_size", "vocab_size", "dtype", "device",
        ]:
            value_label = QLabel("--", group)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addRow(f"{field}:", value_label)
            self.metadata_labels[field] = value_label

        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Sandbox Log", self)
        layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit(group)
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Sandbox activity will appear here...")
        layout.addWidget(self.log_view)
        return group

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    def _on_browse_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select local model folder")
        if folder:
            self.model_path_field.setText(folder)

    def _on_load_clicked(self) -> None:
        target = self.model_path_field.text().strip()
        if not target:
            QMessageBox.warning(self, APP_TITLE, "Enter a model id or local path first.")
            return

        self.append_log(f"Loading '{target}' into sandbox...")

        if self._on_load_model_requested is None:
            self.append_log("(no backend wired up yet -- UI-only demo mode)")
            return

        try:
            metadata = self._on_load_model_requested(target)
        except Exception as exc:  # noqa: BLE001 -- surface any backend error to the user
            self.append_log(f"FAILED: {exc}")
            QMessageBox.critical(self, APP_TITLE, f"Could not load model:\n{exc}")
            return

        self.display_metadata(metadata)
        self.append_log("Model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Public helpers (also used by main.py / tests)
    # ------------------------------------------------------------------ #
    def display_metadata(self, metadata: dict) -> None:
        for field, label in self.metadata_labels.items():
            value = metadata.get(field, "--")
            if field == "num_parameters" and isinstance(value, int):
                value = f"{value:,}"
            label.setText(str(value))

    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
