# Wiring AnomalyReportWidget into main_window.py

I didn't overwrite your existing `ui/main_window.py` here, since I don't
have its current exact contents and a blind full-file replace risks
clobbering the heatmap wiring you already built in Week 2. Instead, add
these ~10 lines by hand (same pattern as your existing "Load Fuzz Sweep
Log..." button):

```python
from ui.anomaly_report_widget import AnomalyReportWidget

# in MainWindow.__init__, near where you set up the heatmap group:
self.anomaly_widget = AnomalyReportWidget()
load_anomaly_button = QPushButton("Load Anomaly Report...")
load_anomaly_button.clicked.connect(self.load_anomaly_report)
# add load_anomaly_button and self.anomaly_widget to a new QGroupBox,
# same as your "Neuron Activation Heatmap" group box

# a new method on MainWindow, mirroring load_sweep_log():
def load_anomaly_report(self):
    path, _ = QFileDialog.getOpenFileName(
        self, "Load Anomaly Report", "outputs/", "JSON files (*.json)"
    )
    if not path:
        return
    try:
        self.anomaly_widget.load_report(path)
    except Exception as e:
        QMessageBox.critical(self, "Error loading report", str(e))
```
