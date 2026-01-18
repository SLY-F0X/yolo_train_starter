import os
import gc
import re
import colorama
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
import ultralytics


def get_base_path() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def get_device_str() -> str:
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def train_model(base_path: Path, device: str):
    ultralytics.checks()

    print("Using device:", device)

    model = YOLO(base_path / "yolo26n.pt")
    model.info()

    model.train(
        data=str(base_path / "data.yaml"),
        batch=-1,
        epochs=300,
        imgsz=640,
        save=True,
        device=device,
        cache=True,
        pretrained=True,
        close_mosaic=0,
        patience=100,
        verbose=True,
    )

    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
        print("Clean CUDA cache")


def _latest_train_dir(runs_path: Path) -> Path | None:
    best = None
    best_num = -1
    pattern = re.compile(r"^train(\d+)?$")

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


def check_model_metrics(base_path: Path, device: str):
    runs_path = base_path / "runs" / "detect"
    if not runs_path.exists():
        print("No runs/detect directory found")
        return

    latest_train_path = _latest_train_dir(runs_path)
    if latest_train_path is None:
        print("No train* directories found")
        return

    model_path = latest_train_path / "weights" / "best.pt"
    print(f"Using model from: {model_path}")

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    model = YOLO(model_path)
    model.info()

    results = model.val(data=str(base_path / "data.yaml"), device=device)

    print("~~~~ Main metrics ~~~~")
    print("mAP50-95:", getattr(results.box, "map", None))
    print("mAP50:", getattr(results.box, "map50", None))
    print("Precision (mean):", getattr(results.box, "mp", None))
    print("Recall (mean):", getattr(results.box, "mr", None))
    print("~~~~ All ~~~~")
    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Average precision:", results.box.ap)
    print("Average precision at IoU=0.50:", results.box.ap50)
    print("Class indices for average precision:", results.box.ap_class_index)
    #print("Class-specific results:", results.box.class_result)
    print("F1 score:", results.box.f1)
    #print("F1 score curve:", results.box.f1_curve)
    #print("Overall fitness score:", results.box.fitness)
    print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision at IoU=0.75:", results.box.map75)
    print("Mean average precision for different IoU thresholds:", results.box.maps)
    #print("Mean results for different metrics:", results.box.mean_results)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    #print("Precision curve:", results.box.p_curve)
    #print("Precision values:", results.box.prec_values)
    #print("Specific precision metrics:", results.box.px)
    print("Recall:", results.box.r)
    #print("Recall curve:", results.box.r_curve)


def export_to_onnx(base_path: Path, device: str):
    runs_path = base_path / "runs" / "detect"
    if not runs_path.exists():
        print("No runs/detect directory found")
        return

    latest_train_path = _latest_train_dir(runs_path)
    if latest_train_path is None:
        print("No train* directories found")
        return

    model_path = latest_train_path / "weights" / "best.pt"
    print(f"Using model from: {model_path}")

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    model = YOLO(model_path)
    model.export(format="onnx", optimize=False, device=device)


def fine_tune_resume(base_path: Path, device: str):
    runs_path = base_path / "runs" / "detect"
    if not runs_path.exists():
        print("No runs/detect directory found")
        return

    latest_train_path = _latest_train_dir(runs_path)
    if latest_train_path is None:
        print("No train* directories found")
        return

    model_path = latest_train_path / "weights" / "best.pt"
    print(f"Using model from: {model_path}")

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    model = YOLO(model_path)
    model.info()
    
    # Resume tuning run with name 'finetune_model' 
    results = model.tune( data=base_path / "data.yaml", epochs=100, iterations=10, optimizer="AdamW", save=True, device=device, name="finetune_model", close_mosaic=0, resume=True)


def _prepare_runtime(device_arg: str | None, set_cuda_index: int | None):
    """
    Возвращает device строку для Ultralytics
    device_arg:
      - None => auto (cuda '0' если доступно, иначе cpu)
      - 'cpu' => cpu
      - '0'/'1'/... => конкретный cuda девайс
    set_cuda_index:
      - None => не трогаем torch.cuda.set_device
      - int  => явно ставим torch.cuda.set_device(int)
    """
    if device_arg is None:
        device = get_device_str()
    else:
        device = device_arg

    print("CUDA available:", torch.cuda.is_available())

    if device != "cpu" and torch.cuda.is_available():
        if set_cuda_index is not None:
            torch.cuda.set_device(set_cuda_index)
        else:
            try:
                torch.cuda.set_device(int(device))
            except (ValueError, TypeError):
                pass

    return device


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ultralytics YOLO training/validation/export helper. "
                    "Без аргументов запускает обучение"
    )

    # Общие опции
    p.add_argument(
        "--device",
        default=None,
        help="Устройство для Ultralytics: 'cpu' или индекс GPU ('0', '1', ...). "
             "По умолчанию авто: '0' если CUDA доступна, иначе 'cpu'."
    )
    p.add_argument(
        "--cuda-index",
        type=int,
        default=None,
        help="Принудительно вызвать torch.cuda.set_device(<index>)."
    )

    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Запустить обучение.")
    sub.add_parser("val", help="Запустить валидацию по последнему train* прогону.")
    sub.add_parser("export", help="Экспорт последнего best.pt в ONNX.")
    sub.add_parser("resume", help="Загрузить последний best.pt и дообучить")

    return p


def main(argv: list[str] | None = None):
    base_path = get_base_path()
    parser = _build_argparser()
    args = parser.parse_args(argv)

    device = _prepare_runtime(args.device, args.cuda_index)

    # Если запуск без аргументов (cmd is None) сразу train.
    if args.cmd is None or args.cmd == "train":
        train_model(base_path, device)
        return

    if args.cmd == "val":
        check_model_metrics(base_path, device)
        return

    if args.cmd == "export":
        export_to_onnx(base_path, device)
        return

    if args.cmd == "resume":
        fine_tune_resume(base_path, device)
        return

    parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    colorama.just_fix_windows_console()
    main()
