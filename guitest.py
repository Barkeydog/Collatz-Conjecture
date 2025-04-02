import sys
import torch
from itertools import product
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gpu_usage():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,nounits,noheader']
        )
        usage, mem = result.decode().strip().split(',')
        return int(usage), int(mem)
    except:
        return 0, 0

class CollatzWorker(QThread):
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(list)
    loop_found = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def run(self):
        max_len = 7
        max_iters = 400
        loss_threshold = 1e-8

        all_ops = []
        for length in range(3, max_len + 1):
            all_ops.extend(list(product(['odd', 'even'], repeat=length)))
        total = len(all_ops)

        for idx, ops in enumerate(all_ops):
            if ops == ("odd", "even", "even"):
                continue

            self.progress_signal.emit(int(100 * idx / total))
            self.log_signal.emit(f"Trying ops: {ops}")

            x = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=0.005)

            loss_data = []
            prev_loss = float("inf")
            stable_count = 0

            for i in range(max_iters):
                optimizer.zero_grad()
                result = x
                for op in ops:
                    result = 3 * result + 1 if op == 'odd' else result / 2
                loss = (result - x).pow(2)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                loss_data.append(current_loss)

                if i % 50 == 0:
                    self.plot_signal.emit(loss_data[:])

                if abs(prev_loss - current_loss) < 1e-12:
                    stable_count += 1
                    if stable_count >= 5:
                        break
                else:
                    stable_count = 0
                prev_loss = current_loss

                if current_loss < loss_threshold:
                    with torch.no_grad():
                        val = x.item()
                        if abs(val - round(val)) < 1e-5 and val > 1:
                            loop_val = round(val)
                            sequence = []
                            n = loop_val
                            for op in ops:
                                sequence.append(n)
                                n = 3 * n + 1 if op == 'odd' else n // 2
                            sequence.append(n)
                            if sequence[-1] == sequence[0]:
                                if set(sequence).issubset({1, 2, 4}):
                                    continue
                                msg = f"\n🚨 Found non-trivial loop!\nStart: {loop_val}\nSequence: {sequence}"
                                self.loop_found.emit(msg)
                                return

        self.progress_signal.emit(100)
        self.log_signal.emit("\n✅ Done. No non-trivial loop found.")

class CollatzGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collatz Loop Finder (Optimized + Progress Bar)")
        self.setGeometry(200, 200, 700, 650)

        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.gpu_label = QLabel("GPU Usage: --%, Mem: --MB")
        layout.addWidget(self.gpu_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.canvas = FigureCanvas(Figure(figsize=(6, 3)))
        self.ax = self.canvas.figure.subplots()
        layout.addWidget(self.canvas)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_worker)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

        self.worker = None

        self.gpu_timer = QTimer()
        self.gpu_timer.timeout.connect(self.update_gpu)
        self.gpu_timer.start(1000)

    def log(self, msg):
        self.log_box.append(msg)

    def update_plot(self, losses):
        self.ax.clear()
        self.ax.plot(losses)
        self.ax.set_title("Loss Over Time")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Loss")
        self.canvas.draw()

    def update_gpu(self):
        usage, mem = get_gpu_usage()
        self.gpu_label.setText(f"GPU Usage: {usage}%, Mem: {mem}MB")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def start_worker(self):
        if self.worker is None or not self.worker.isRunning():
            self.log("Starting search...")
            self.status_label.setText("Status: Running")
            self.progress_bar.setValue(0)

            self.worker = CollatzWorker()
            self.worker.log_signal.connect(self.log)
            self.worker.plot_signal.connect(self.update_plot)
            self.worker.loop_found.connect(self.loop_found)
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.start()

    def loop_found(self, msg):
        self.log(msg)
        self.status_label.setText("Status: Loop Found ✅")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CollatzGUI()
    gui.show()
    sys.exit(app.exec_())
