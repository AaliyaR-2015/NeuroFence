"""
main.py
-------
Entry point for the NeuroFence desktop app. Wires the PyQt UI skeleton
(ui/main_window.py) to the real model sandbox loader (src/model_sandbox.py).

Run with:
    python main.py
"""

import sys

from PyQt5.QtWidgets import QApplication

from src.model_sandbox import load_model_safely
from ui.main_window import MainWindow


def load_model_for_ui(target: str) -> dict:
    """
    Adapter between the UI (which wants a plain dict) and the sandbox loader
    (which returns model/tokenizer/ModelMetadata). The UI never touches
    torch/transformers objects directly -- it only ever sees metadata.
    """
    _model, _tokenizer, metadata = load_model_safely(target)
    return metadata.as_dict()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow(on_load_model_requested=load_model_for_ui)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
