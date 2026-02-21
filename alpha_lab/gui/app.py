from __future__ import annotations

import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from alpha_lab.pipeline import run_pipeline

try:
    from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime import guard
    raise RuntimeError("PyQt6 is required to run GUI mode.") from exc


@dataclass(slots=True)
class UiState:
    report_path: Path | None = None


class WorkerSignals(QObject):
    started = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class RunWorker(QRunnable):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.signals = WorkerSignals()

    def run(self) -> None:
        self.signals.started.emit(f"Running task with config: {self.config_path}")
        try:
            result = run_pipeline(self.config_path)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Alpha-Lab")
        self.resize(1080, 720)
        self.state = UiState()
        pool = QThreadPool.globalInstance()
        self.thread_pool = pool if pool is not None else QThreadPool(self)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        config_tab = QWidget()
        config_layout = QFormLayout(config_tab)
        self.config_input = QLineEdit("configs/example.yaml")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_config)
        config_row = QHBoxLayout()
        config_row.addWidget(self.config_input)
        config_row.addWidget(browse_btn)
        config_widget = QWidget()
        config_widget.setLayout(config_row)
        config_layout.addRow("Task Config", config_widget)

        self.run_btn = QPushButton("Run Task")
        self.run_btn.clicked.connect(self._run_task)
        config_layout.addRow(self.run_btn)
        tabs.addTab(config_tab, "Task Config")

        run_tab = QWidget()
        run_layout = QVBoxLayout(run_tab)
        self.status_label = QLabel("Status: idle")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        run_layout.addWidget(self.status_label)
        run_layout.addWidget(self.log_box)
        tabs.addTab(run_tab, "Task Runner")

        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        self.report_label = QLabel("Report: -")
        report_layout.addWidget(self.report_label)
        report_btn_row = QHBoxLayout()
        self.open_report_btn = QPushButton("Open Report")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.clicked.connect(self._open_report)
        report_btn_row.addWidget(self.open_report_btn)
        report_btn_row.addStretch(1)
        report_layout.addLayout(report_btn_row)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        report_layout.addWidget(self.metrics_table)
        tabs.addTab(report_tab, "Report Viewer")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f7f9fc; }
            QTabWidget::pane {
                border: 1px solid #dce3ef; background: #ffffff; border-radius: 6px;
            }
            QTabBar::tab {
                background: #eaf0fb; color: #23364d; padding: 8px 14px; margin-right: 2px;
            }
            QTabBar::tab:selected { background: #d8e5fb; font-weight: 600; }
            QPushButton {
                background: #2d5da8;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 7px 14px;
            }
            QPushButton:disabled { background: #9bb2d7; }
            QLineEdit, QPlainTextEdit, QTableWidget {
                border: 1px solid #d2d9e5;
                border-radius: 5px;
                background: #ffffff;
            }
            QLabel { color: #223249; }
            """
        )

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select config",
            "",
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.config_input.setText(path)

    def _append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _run_task(self) -> None:
        config_path = Path(self.config_input.text().strip())
        if not config_path.exists():
            QMessageBox.warning(
                self,
                "Invalid Config",
                f"Config file does not exist: {config_path}",
            )
            return

        self.run_btn.setEnabled(False)
        self.status_label.setText("Status: running")
        self._append_log(f"[INFO] Start task with {config_path}")

        worker = RunWorker(config_path)
        worker.signals.started.connect(self._append_log)
        worker.signals.finished.connect(self._on_task_finished)
        worker.signals.failed.connect(self._on_task_failed)
        self.thread_pool.start(worker)

    def _on_task_finished(self, result: object) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Status: completed")
        self._append_log("[INFO] Task completed successfully.")

        report_path = getattr(result, "report_path", None)
        if isinstance(report_path, Path):
            self.state.report_path = report_path
            self.report_label.setText(f"Report: {report_path}")
            self.open_report_btn.setEnabled(True)
        else:
            self.report_label.setText("Report: not generated")

        self._fill_metrics_table(result)

    def _fill_metrics_table(self, result: object) -> None:
        evaluations = getattr(result, "evaluation", {})
        if not evaluations:
            self.metrics_table.setRowCount(0)
            return
        first_factor = next(iter(evaluations.values()))
        metrics = getattr(first_factor, "metrics", {})
        self.metrics_table.setRowCount(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(k)))
            value = f"{v:.6f}" if isinstance(v, float) else str(v)
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        self.metrics_table.resizeColumnsToContents()

    def _on_task_failed(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Status: failed")
        self._append_log("[ERROR] Task failed.")
        self._append_log(err)
        QMessageBox.critical(
            self,
            "Task Failed",
            err.splitlines()[-1] if err else "Unknown error",
        )

    def _open_report(self) -> None:
        if self.state.report_path is None:
            return
        webbrowser.open(self.state.report_path.resolve().as_uri())


def launch_gui() -> int:
    app = QApplication([])
    app.setApplicationName("Alpha-Lab")
    window = MainWindow()
    window.show()
    return app.exec()
