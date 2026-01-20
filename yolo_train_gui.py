import threading
import os
import subprocess
import platform
import time
import re
from typing import Iterable
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont

import yolo_train_starter as yts


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


def _latest_run_dir(runs_path: Path, prefix: str) -> Path | None:
    """
    Search for: prefix, prefix2, prefix3 ...
    """
    best = None
    best_num = -1
    # train, train2, train3... | val, val2, val3...
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)?$")

    for d in runs_path.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        num = int(m.group(1)) if m.group(1) else 1
        if num > best_num:
            best_num = num
            best = d

    return best


def _newest_by_mtime(paths: Iterable[Path]) -> Path | None:
    best = None
    best_ts = -1.0
    for p in paths:
        try:
            ts = p.stat().st_mtime
        except OSError:
            continue
        if ts > best_ts:
            best_ts = ts
            best = p
    return best


class YoloTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("YOLO Train Starter GUI")
        self.geometry("680x340")

        self.worker_thread = None
        self.base_path = yts.get_base_path()

        self._build_styles_and_fonts()
        self._build_ui()

        self._append_text(f"Base path: {self.base_path}\n")
        self._append_text("GUI ready\n")

        self._refresh_devices(autoselect=True, log=True)
        self._check_cuda(silent=True, refresh_devices=False)

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

        self.log_font = tkfont.Font(family=mono_family, size=10)

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
            width=10,
        )
        self.device_combo.pack(side=tk.LEFT, padx=(6, 16))

        self.cuda_info_var = tk.StringVar(value="CUDA available: (unknown)")
        ttk.Label(top, textvariable=self.cuda_info_var).pack(side=tk.LEFT)

        ttk.Button(top, text="Refresh devices", command=self._refresh_devices).pack(
            side=tk.LEFT, padx=(16, 6)
        )
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

        self.log_text.configure(state=tk.DISABLED)

    def _ui_log(self, s: str):
        self.after(0, self._append_text, s)

    def _refresh_devices(self, autoselect: bool = False, log: bool = True):
        try:
            import torch

            values = ["cpu"]
            avail = torch.cuda.is_available()
            n = torch.cuda.device_count() if avail else 0

            if avail and n > 0:
                values.extend([str(i) for i in range(n)])

            current_values = list(self.device_combo["values"]) if self.device_combo["values"] else ["cpu"]
            if len(values) > 1 or current_values == ["cpu"]:
                self.device_combo["values"] = values
            else:
                values = current_values

            current = self.device_var.get().strip()

            if autoselect and current == "cpu" and "0" in values:
                self.device_var.set("0")
                current = "0"

            if current not in values:
                self.device_var.set("cpu")

            if log:
                self._append_text(f"Devices refreshed: {list(values)}\n")

        except Exception as e:
            self._append_text(f"Refresh devices error: {e}\n")
            if not self.device_combo["values"]:
                self.device_combo["values"] = ["cpu"]
            if self.device_var.get().strip() not in self.device_combo["values"]:
                self.device_var.set("cpu")


    def _check_cuda(self, silent: bool = False, refresh_devices: bool = True):
        try:
            import torch

            avail = torch.cuda.is_available()
            if avail:
                try:
                    _ = torch.cuda.current_device()
                    _ = torch.cuda.get_device_name(_)
                except Exception:
                    avail = False

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

            if refresh_devices:
                self._refresh_devices(autoselect=True, log=not silent)

        except Exception as e:
            self.cuda_info_var.set("CUDA available: error")
            self._append_text(f"CUDA check error: {e}\n")


    def _collect_artifacts(self, label: str) -> str:
        runs_path = self.base_path / "runs" / "detect"
        if not runs_path.exists():
            return "Artifacts: runs/detect not found.\n"

        lines = []
        lines.append("Artifacts:\n")

        label_l = label.lower()

        # TRAIN
        if "train" in label_l:
            train_dir = _latest_run_dir(runs_path, "train")
            if not train_dir:
                return "Artifacts: no train* dirs found.\n"

            lines.append(f"  run dir: {train_dir}\n")

            wdir = train_dir / "weights"
            best_pt = wdir / "best.pt"
            last_pt = wdir / "last.pt"
            if best_pt.exists():
                lines.append(f"  best.pt: {best_pt}\n")
            if last_pt.exists():
                lines.append(f"  last.pt: {last_pt}\n")

            for name in ("results.csv", "results.png", "args.yaml"):
                p = train_dir / name
                if p.exists():
                    lines.append(f"  {name}: {p}\n")

            return "".join(lines)

        # VALIDATE
        if "validate" in label_l or "val" in label_l:
            val_dir = _latest_run_dir(runs_path, "val")
            if not val_dir:
                return "Artifacts: no val* dirs found.\n"

            lines.append(f"  val dir: {val_dir}\n")

            common = [
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
            ]
            for name in common:
                p = val_dir / name
                if p.exists():
                    lines.append(f"  {name}: {p}\n")

            return "".join(lines)

        # EXPORT
        if "export" in label_l and "onnx" in label_l:
            train_dir = _latest_run_dir(runs_path, "train")
            if not train_dir:
                return "Artifacts: no train* dirs found (can't locate best.pt for export).\n"

            wdir = train_dir / "weights"
            best_pt = wdir / "best.pt"
            lines.append(f"  weights dir: {wdir}\n")
            if best_pt.exists():
                lines.append(f"  source: {best_pt}\n")

            onnx_files = list(wdir.glob("*.onnx"))
            newest = _newest_by_mtime(onnx_files)
            if newest:
                lines.append(f"  onnx: {newest}\n")
            else:
                expected = best_pt.with_suffix(".onnx")
                if expected.exists():
                    lines.append(f"  onnx: {expected}\n")
                else:
                    lines.append("  onnx: not found in weights dir.\n")

            return "".join(lines)

        # RESUME / TUNE
        if "fine-tune" in label_l or "resume" in label_l or "tune" in label_l:
            candidates = []
            for d in runs_path.iterdir():
                if d.is_dir() and d.name.startswith("finetune_model"):
                    candidates.append(d)
            finedir = _newest_by_mtime(candidates) if candidates else None

            if finedir:
                lines.append(f"  tune dir: {finedir}\n")
                wdir = finedir / "weights"
                if wdir.exists():
                    best_pt = wdir / "best.pt"
                    last_pt = wdir / "last.pt"
                    if best_pt.exists():
                        lines.append(f"  best.pt: {best_pt}\n")
                    if last_pt.exists():
                        lines.append(f"  last.pt: {last_pt}\n")
                return "".join(lines)

            train_dir = _latest_run_dir(runs_path, "train")
            if train_dir:
                lines.append(f"  (fallback) latest train dir: {train_dir}\n")
                return "".join(lines)

            return "Artifacts: nothing found.\n"

        return "Artifacts: (no collector rule for this action)\n"


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
            self._ui_log(f"\n--- {label} started ---\n")
            try:
                target(base_path, device)
                self._ui_log(f"--- {label} finished ---\n")

                try:
                    artifacts_text = self._collect_artifacts(label)
                    self._ui_log(artifacts_text)
                except Exception as e:
                    self._ui_log(f"Artifacts collection error: {e}\n")

            except Exception as e:
                self._ui_log(f"--- {label} failed: {e} ---\n")
            finally:
                self.after(0, self._set_buttons_enabled, True)

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
        self.log_text.insert(tk.END, s)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main():
    app = YoloTrainerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
