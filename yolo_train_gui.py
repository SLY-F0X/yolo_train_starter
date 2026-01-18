import sys
import threading
import queue
import logging
import os
import subprocess
import platform
import re
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from colorama import init as colorama_init, deinit as colorama_deinit

import yolo_train_starter as yts

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # стандартные ANSI CSI
ANSI_OSC_RE = re.compile(r"\x1b\].*?\x07")
CTRL_OTHER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def strip_ansi(s: str) -> str:
    s = ANSI_OSC_RE.sub("", s)
    s = ANSI_RE.sub("", s)
    return s


def normalize_stream_text(s: str) -> str:
    s = strip_ansi(s)
    s = CTRL_OTHER_RE.sub("", s)
    s = s.replace("\r", "\n")
    return s


def _is_emoji_char(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)

    if o == 0x200D:
        return True

    if 0x1F000 <= o <= 0x1FAFF:
        return True
    if 0x2600 <= o <= 0x27BF:
        return True
    if 0xFE00 <= o <= 0xFE0F:
        return True
    if 0x1F1E6 <= o <= 0x1F1FF:
        return True
    return False


def _emoji_spans(s: str):
    spans = []
    start = None
    for i, ch in enumerate(s):
        is_e = _is_emoji_char(ch)
        if is_e and start is None:
            start = i
        elif (not is_e) and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(s)))
    return spans


class QueueWriter:
    def __init__(self, q: queue.Queue, stream_name: str):
        self.q = q
        self.stream_name = stream_name

    def write(self, data):
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8", errors="replace")
            except Exception:
                data = data.decode(errors="replace")
        if not isinstance(data, str):
            data = str(data)
        if not data:
            return

        try:
            self.q.put_nowait((self.stream_name, data))
        except queue.Full:
            pass

    def flush(self):
        pass



class LoggingToQueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.log_queue = queue.Queue(maxsize=50000)

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.q.put(("log", msg + "\n"))


def open_folder_cross_platform(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    system = platform.system().lower()
    if system.startswith("windows"):
        os.startfile(str(path))
    elif system == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class YoloTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("YOLO Train Starter GUI")
        self.geometry("980x640")

        self.log_queue = queue.Queue()
        self.worker_thread = None

        self.base_path = yts.get_base_path()

        self._build_styles_and_fonts()
        self._build_ui()

        self.after(50, self._drain_log_queue)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = QueueWriter(self.log_queue, "stdout")
        sys.stderr = QueueWriter(self.log_queue, "stderr")

        self._install_logging_handler()

        self._append_text(f"Base path: {self.base_path}\n")
        self._append_text("GUI ready ✅\n")

        self._refresh_devices()
        self._check_cuda(silent=True)

    def _build_styles_and_fonts(self):
        families = set(tkfont.families(self))

        if "Consolas" in families:
            mono_family = "Consolas"
        elif "DejaVu Sans Mono" in families:
            mono_family = "DejaVu Sans Mono"
        elif "Liberation Mono" in families:
            mono_family = "Liberation Mono"
        else:
            mono_family = "TkFixedFont"

        if "Segoe UI Emoji" in families:
            emoji_family = "Segoe UI Emoji"
        elif "Noto Color Emoji" in families:
            emoji_family = "Noto Color Emoji"
        elif "Apple Color Emoji" in families:
            emoji_family = "Apple Color Emoji"
        else:
            emoji_family = mono_family

        self.log_font = tkfont.Font(family=mono_family, size=10)
        self.emoji_font = tkfont.Font(family=emoji_family, size=10)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Device:").pack(side=tk.LEFT)

        self.device_var = tk.StringVar(value="cpu")
        self.device_combo = ttk.Combobox(
            top,
            textvariable=self.device_var,
            values=["cpu"],
            state="readonly",
            width=10
        )
        self.device_combo.pack(side=tk.LEFT, padx=(6, 16))

        self.cuda_info_var = tk.StringVar(value="CUDA available: (unknown)")
        ttk.Label(top, textvariable=self.cuda_info_var).pack(side=tk.LEFT)

        ttk.Button(top, text="Refresh devices", command=self._refresh_devices).pack(side=tk.LEFT, padx=(16, 6))
        ttk.Button(top, text="Check CUDA", command=self._check_cuda).pack(side=tk.LEFT)

        actions = ttk.Frame(self, padding=(10, 0, 10, 10))
        actions.pack(side=tk.TOP, fill=tk.X)

        self.btn_train = ttk.Button(actions, text="Train model", command=self._on_train)
        self.btn_train.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_val = ttk.Button(actions, text="Validate", command=self._on_validate)
        self.btn_val.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_export = ttk.Button(actions, text="Export to ONNX", command=self._on_export)
        self.btn_export.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_resume = ttk.Button(actions, text="Fine-tune (resume stub)", command=self._on_resume)
        self.btn_resume.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_open_runs = ttk.Button(actions, text="Open runs/detect", command=self._open_runs_detect)
        self.btn_open_runs.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ttk.Button(actions, text="Clear log", command=self._clear_log)
        self.btn_clear.pack(side=tk.RIGHT)

        log_frame = ttk.Frame(self, padding=10)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, font=self.log_font)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.log_text.tag_configure("emoji", font=self.emoji_font)

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, "")
        self.log_text.configure(state=tk.DISABLED)

    def _install_logging_handler(self):
        handler = LoggingToQueueHandler(self.log_queue)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        logging.getLogger("ultralytics").setLevel(logging.INFO)

    def _refresh_devices(self):
        try:
            import torch

            values = ["cpu"]
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                for i in range(n):
                    values.append(str(i))

            current = self.device_var.get().strip()
            self.device_combo["values"] = values

            if current not in values:
                self.device_var.set("cpu")

            self._append_text(f"Devices refreshed: {values}\n")
        except Exception as e:
            self._append_text(f"Refresh devices error: {e}\n")
            self.device_combo["values"] = ["cpu"]
            self.device_var.set("cpu")

    def _check_cuda(self, silent: bool = False):
        try:
            import torch

            avail = torch.cuda.is_available()
            n = torch.cuda.device_count() if avail else 0
            self.cuda_info_var.set(f"CUDA available: {avail} | GPUs: {n}")

            if not silent:
                self._append_text(f"CUDA available: {avail}\n")
                self._append_text(f"CUDA device_count: {n}\n")

            dev = self.device_var.get().strip()
            if avail and dev != "cpu":
                try:
                    torch.cuda.set_device(int(dev))
                    if not silent:
                        name = torch.cuda.get_device_name(int(dev))
                        self._append_text(f"CUDA device set to {dev} ({name})\n")
                except Exception as e:
                    self._append_text(f"Failed to set CUDA device {dev}: {e}\n")
                    self.device_var.set("cpu")
            self._refresh_devices()

        except Exception as e:
            self.cuda_info_var.set("CUDA available: error")
            self._append_text(f"CUDA check error: {e}\n")

    def _open_runs_detect(self):
        runs_detect = self.base_path / "runs" / "detect"
        try:
            if not runs_detect.exists():
                self._append_text(f"Folder not found: {runs_detect}\n")
                messagebox.showinfo("Info", f"Folder does not exist:\n{runs_detect}")
                return
            open_folder_cross_platform(runs_detect)
            self._append_text(f"Opened folder: {runs_detect}\n")
        except Exception as e:
            self._append_text(f"Open folder error: {e}\n")
            messagebox.showerror("Error", f"Failed to open folder:\n{e}")

    def _set_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for b in (self.btn_train, self.btn_val, self.btn_export, self.btn_resume, self.btn_open_runs):
            b.configure(state=state)

    def _run_in_thread(self, target, label: str):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        device = self.device_var.get().strip()
        base_path = self.base_path

        def runner():
            self.log_queue.put(("stdout", f"\n--- {label} started ---\n"))
            try:
                target(base_path, device)
                self.log_queue.put(("stdout", f"--- {label} finished ✅ ---\n"))
            except Exception as e:
                self.log_queue.put(("stderr", f"--- {label} failed ❌: {e} ---\n"))
            finally:
                self.log_queue.put(("ui", "ENABLE_BUTTONS"))

        self._set_buttons_enabled(False)
        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()

    def _on_train(self):
        self._run_in_thread(yts.train_model, "Train model")

    def _on_validate(self):
        self._run_in_thread(yts.check_model_metrics, "Validate")

    def _on_export(self):
        self._run_in_thread(yts.export_to_onnx, "Export to ONNX")

    def _on_resume(self):
        self._run_in_thread(yts.fine_tune_resume, "Fine-tune resume")

    def _clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _append_text(self, s: str):
        self.log_text.configure(state=tk.NORMAL)

        start_index = self.log_text.index(tk.END)
        self.log_text.insert(tk.END, s)

        spans = _emoji_spans(s)
        for a, b in spans:
            idx_a = f"{start_index}+{a}c"
            idx_b = f"{start_index}+{b}c"
            self.log_text.tag_add("emoji", idx_a, idx_b)

        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


    def _drain_log_queue(self):
        start = time.monotonic()

        MAX_ITEMS = 500
        MAX_SECONDS = 0.01

        chunks = []
        ui_enable = False

        items = 0
        try:
            while items < MAX_ITEMS and (time.monotonic() - start) < MAX_SECONDS:
                stream, data = self.log_queue.get_nowait()

                if stream == "ui" and data == "ENABLE_BUTTONS":
                    ui_enable = True
                    items += 1
                    continue

                if isinstance(data, bytes):
                    try:
                        data = data.decode("utf-8", errors="replace")
                    except Exception:
                        data = data.decode(errors="replace")
                if not isinstance(data, str):
                    data = str(data)

                data = normalize_stream_text(data)
                if data:
                    chunks.append(data)

                items += 1

        except queue.Empty:
            pass
        finally:
            if chunks:
                self._append_text("".join(chunks))

            if ui_enable:
                self._set_buttons_enabled(True)

            self.after(50, self._drain_log_queue)


    def destroy(self):
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        super().destroy()


def main():
    colorama_init(autoreset=True, strip=True, convert=False)

    try:
        app = YoloTrainerGUI()
        app.mainloop()
    finally:
        colorama_deinit()

if __name__ == "__main__":
    main()