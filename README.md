# yolo_train_starter

Набор файлов для быстрого старта обучения YOLO (Ultralytics) на своём датасете. 

Репозиторий ориентирован на Windows и установку зависимостей через **uv** (виртуальное окружение + lock-файл).

## Что внутри

- `yolo_train_starter.py` — запуск обучения (CLI-скрипт).
- `yolo_train_gui.py` — GUI для запуска обучения (запускается батником).
- `yolo_class_tools.py` — вспомогательные утилиты (подготовка/проверка классов и т.п.).
- `data.yaml` — конфиг датасета в формате Ultralytics.
- `yolo26n.pt` — стартовые веса (pretrained) для fine-tuning.
- Батники:
  - `Setup_uv_venv.bat` — установка uv, создание `.venv`, установка PyTorch CUDA 13.0 (cu130) и синхронизация зависимостей.
  - `Start_uv.bat` — открыть консоль уже активированного окружения.
  - `Run_train.bat` — запуск обучения.
  - `Run_GUI.bat` — запуск GUI.
- Папки датасета:
  - `images/train`, `images/val` — изображения
  - `labels/train`, `labels/val` — разметка YOLO (txt)

## Требования

- Windows 10
- Python **3.12** (см. `.python-version` и `pyproject.toml`).
- NVIDIA GPU и CUDA драйвер (проект ставит `torch/torchvision/torchaudio` с backend **cu130**).
- Если GPU/драйвер не подходят, придётся заменить установку PyTorch на CPU-версию (см. раздел `Решение проблем`).

## Структура датасета

Стандартная структура Ultralytics:

```
yolo_train_starter/
  data.yaml
  images/
    train/
      *.jpg|*.png|...
    val/
      *.jpg|*.png|...
  labels/
    train/
      *.txt
    val/
      *.txt
```

Файлы разметки в `labels/*/*.txt` — формат YOLO:
- по одной строке на объект
- `class_id x_center y_center width height`
- координаты нормализованы в диапазоне 0..1

Пример строки:
```
0 0.512 0.423 0.120 0.210
```

## Конфиг датасета: `data.yaml`

Текущий `data.yaml`:

```yaml
train: ../images/train
val: ../images/val

names:
  0: Object
```

Важно:
- Пути `train/val` указаны **относительно файла `data.yaml`**. Здесь задано `../images/...`, т.е. ожидается, что `data.yaml` лежит в `yolo_train_starter/`, а `images/` — рядом с ним (как в структуре проекта).
- Сейчас определён **один** класс: `0: Object`. Если классов больше — добавьте их по порядку.

## Быстрый старт (Windows)

1) Склонируйте/распакуйте проект в папку без кириллицы и пробелов в пути.

2) Подготовьте датасет:
- положите изображения в `images/train` и `images/val`
- положите соответствующие txt-разметки в `labels/train` и `labels/val`
- при необходимости обновите `data.yaml` и `names`

3) Создайте окружение и установите зависимости:

Запустите:
```
Setup_uv_venv.bat
```

Либо вручную:
``` CMD
pip install -U uv
uv venv --python 3.12 --seed --no-cache-dir --relocatable --allow-existing --link-mode=copy --prompt "Yolo-Env"
set UV_HTTP_TIMEOUT=600
set HTTP_TIMEOUT=600
uv pip install torch torchvision torchaudio --torch-backend=cu130 --no-cache-dir --link-mode=copy
uv sync --index https://download.pytorch.org/whl/cu130 --no-cache-dir --link-mode=copy
```
4) Запуск обучения:

```
Run_train.bat
```

5) Запуск GUI:

```
Run_GUI.bat
```

6) Открыть консоль с активированным окружением:

```
Start_uv.bat
```

## Зависимости

Основные зависимости описаны в `pyproject.toml` и зафиксированы в `uv.lock`, включая:

- `ultralytics`
- `torch`, `torchvision`, `torchaudio` (через индекс PyTorch cu130)
- `onnx`, `onnxruntime-gpu`, `onnxslim`
- `colorama`

`requirements.txt` сгенерирован uv

## Где менять классы

- В `data.yaml` — список `names`.
- В разметке (`labels/.../*.txt`) — `class_id` должен соответствовать ключам из `names`.

Если меняете число классов, убедитесь, что:
- нет ID, выходящих за пределы `names`
- все разметки соответствуют актуальному набору классов

## Решение проблем

- `[ERROR] .venv not exist` при запуске `Run_train.bat`/`Run_GUI.bat`  
  Значит окружение не создано. Запустите `Setup_uv_venv.bat` и убедитесь, что появилась папка `.venv`.

- Ошибка PyTorch/CUDA (не находит CUDA, несовместим драйвер, и т.п.)  
  Набор скриптов ставит PyTorch с CUDA backend `cu130`. Если GPU/драйвер не поддерживают:
  - поставьте CPU-версии `torch/torchvision/torchaudio`, либо
  - подберите другой CUDA backend/индекс под ваш драйвер.
  
  Быстрый вариант (CPU) — удалить `.venv` и создать окружение заново, затем установить `torch` без `--torch-backend=cu130` и без индекса cu130.

- Датасет "не обнаружен" / 0 изображений  
  Проверьте:
  - пути в `data.yaml`
  - что в `images/train` и `images/val` лежат изображения
  - что разметка лежит в `labels/...` и имена файлов совпадают (например `img_001.jpg` ↔ `img_001.txt`)
