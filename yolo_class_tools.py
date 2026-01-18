import os
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


TR = {
    "ru": {
        "title": "YOLO class tools",
        "lang": "Язык",
        "ru": "RU",
        "en": "EN",

        "tab_folders": "Папки",
        "tab_rename": "Переименование",
        "tab_classes": "Классы",

        "images_folder": "Изображения",
        "labels_folder": "Разметка (.txt)",
        "browse": "Выбрать…",
        "open_log": "Лог",

        "folders_header": "Анализ папок",
        "analyze_folders_btn": "Анализ папок",
        "images_count": "Изображений",
        "labels_count": "Лэйблов",
        "orphans_count": "Лэйблов без изображения",
        "delete_orphans_btn": "Удалить лэйблы без изображений",
        "delete_orphans_confirm": "Найдено лэйблов без изображения: {n}\nУдалить их?",
        "delete_orphans_done": "Удалено файлов: {n}",
        "no_orphans": "Лэйблов без изображения не найдено.",

        "rename_header": "Переименование пар (image+label) по совпадению названий",
        "new_base": "Новое имя",
        "start_index": "Начальный индекс",
        "dry_run": "Проверить без именения файлов",
        "rename_btn": "Переименовать",

        "check_header": "Проверка колличества классов (YOLO)",
        "expected_classes": "Максимальное число ожидаемых классов",
        "analyze_btn": "Анализ",

        "edit_header": "Изменение классов в разметке",
        "new_classes": "Новый диапазон (0..N-1)",
        "edit_dry_run": "Dry-run (не писать файлы)",

        "simple_reassign": "Простое переназначение (один класс)",
        "old_class": "Старый класс",
        "new_class": "Новый класс",
        "apply_reassign": "Применить",

        "mapping": "Маппинг",
        "mapping_rules": "Правила: old->new или old:new (по строкам). Пример:\n76->0\n3:1\nМожно “сливать” классы: 3:1 и 7:1",
        "apply_mapping": "Применить маппинг",

        "err": "Ошибка",
        "ok": "Готово",
        "need_images": "Не выбрана папка с изображениями.",
        "need_labels": "Не выбрана папка с разметкой.",
        "need_base": "Введите новое базовое имя.",
        "no_txt": "В папке разметки нет .txt файлов.",
        "rename_done": "Переименование завершено.",
        "analyze_done": "Анализ завершён.",
        "edit_done": "Изменения применены.",
        "dry_note": "Dry-run: файлы не менялись.",
    },
    "en": {
        "title": "YOLO class tools",
        "lang": "Language",
        "ru": "RU",
        "en": "EN",

        "tab_folders": "Folders",
        "tab_rename": "Rename",
        "tab_classes": "Classes",

        "images_folder": "Images",
        "labels_folder": "Labels (.txt)",
        "browse": "Browse…",
        "open_log": "Log",

        "folders_header": "Folder analysis",
        "analyze_folders_btn": "Analyze folders",
        "images_count": "Images",
        "labels_count": "Labels",
        "orphans_count": "Labels without image",
        "delete_orphans_btn": "Delete orphan labels",
        "delete_orphans_confirm": "Found labels without image: {n}\nDelete them?",
        "delete_orphans_done": "Deleted files: {n}",
        "no_orphans": "No orphan labels found.",

        "rename_header": "Rename pairs (image+label) by matching stem",
        "new_base": "New name",
        "start_index": "Start index",
        "dry_run": "Dry-run",
        "rename_btn": "Rename",

        "check_header": "Class count check (YOLO)",
        "expected_classes": "Expected maximum classes",
        "analyze_btn": "Analyze",

        "edit_header": "Edit classes in labels",
        "new_classes": "New range (0..N-1)",
        "edit_dry_run": "Dry-run (don’t write files)",

        "simple_reassign": "Simple reassignment (single class)",
        "old_class": "Old class",
        "new_class": "New class",
        "apply_reassign": "Apply",

        "mapping": "Mapping (multiple rules, optional)",
        "mapping_rules": "Rules: old->new or old:new (one per line). Example:\n76->0\n3:1\nYou can “merge” classes: 3:1 and 7:1",
        "apply_mapping": "Apply mapping",

        "err": "Error",
        "ok": "Done",
        "need_images": "Images folder not selected.",
        "need_labels": "Labels folder not selected.",
        "need_base": "Enter new base name.",
        "no_txt": "No .txt files in labels folder.",
        "rename_done": "Renaming completed.",
        "analyze_done": "Analysis completed.",
        "edit_done": "Changes applied.",
        "dry_note": "Dry-run: files were not modified.",
    }
}


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master

        self.lang = tk.StringVar(value="ru")

        self.images_dir = tk.StringVar(value="")
        self.labels_dir = tk.StringVar(value="")

        self.new_base = tk.StringVar(value="")
        self.start_index = tk.IntVar(value=1)
        self.rename_dry = tk.BooleanVar(value=False)

        self.expected_classes = tk.IntVar(value=80)

        self.new_classes = tk.IntVar(value=80)
        self.edit_dry = tk.BooleanVar(value=False)

        self.old_class = tk.IntVar(value=1)
        self.new_class = tk.IntVar(value=0)

        # folder analysis state
        self.images_count_var = tk.StringVar(value="0")
        self.labels_count_var = tk.StringVar(value="0")
        self.orphans_count_var = tk.StringVar(value="0")
        self._last_orphan_paths: list[Path] = []

        self.log_window = None
        self.log_text = None

        self._build_ui()
        self._apply_i18n()

    def t(self, k: str) -> str:
        return TR[self.lang.get()].get(k, k)

    def _build_ui(self):
        self.master.title("YOLO GUI")
        self.pack(fill="both", expand=True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top.columnconfigure(3, weight=1)

        self.lbl_lang = ttk.Label(top, text="Language")
        self.lbl_lang.grid(row=0, column=0, padx=(0, 6), sticky="w")

        self.rb_ru = ttk.Radiobutton(top, variable=self.lang, value="ru", command=self._apply_i18n)
        self.rb_en = ttk.Radiobutton(top, variable=self.lang, value="en", command=self._apply_i18n)
        self.rb_ru.grid(row=0, column=1, padx=4, sticky="w")
        self.rb_en.grid(row=0, column=2, padx=4, sticky="w")

        self.btn_log = ttk.Button(top, command=self.open_log)
        self.btn_log.grid(row=0, column=4, sticky="e")

        nb = ttk.Notebook(self)
        nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.tab_folders = ttk.Frame(nb)
        self.tab_rename = ttk.Frame(nb)
        self.tab_classes = ttk.Frame(nb)
        nb.add(self.tab_folders, text="Folders")
        nb.add(self.tab_rename, text="Rename")
        nb.add(self.tab_classes, text="Classes")

        self._build_folders_tab()
        self._build_rename_tab()
        self._build_classes_tab()

    def _build_folders_tab(self):
        f = self.tab_folders
        f.columnconfigure(1, weight=1)

        self.lbl_images = ttk.Label(f, text="Images")
        self.lbl_images.grid(row=0, column=0, padx=8, pady=10, sticky="w")
        ttk.Entry(f, textvariable=self.images_dir).grid(row=0, column=1, padx=8, pady=10, sticky="ew")
        self.btn_images = ttk.Button(f, command=self.pick_images)
        self.btn_images.grid(row=0, column=2, padx=8, pady=10, sticky="ew")

        self.lbl_labels = ttk.Label(f, text="Labels")
        self.lbl_labels.grid(row=1, column=0, padx=8, pady=10, sticky="w")
        ttk.Entry(f, textvariable=self.labels_dir).grid(row=1, column=1, padx=8, pady=10, sticky="ew")
        self.btn_labels = ttk.Button(f, command=self.pick_labels)
        self.btn_labels.grid(row=1, column=2, padx=8, pady=10, sticky="ew")

        # Folder analysis
        self.lbl_folders_header = ttk.Label(f, text="Folder analysis", font=("TkDefaultFont", 10, "bold"))
        self.lbl_folders_header.grid(row=2, column=0, columnspan=3, padx=8, pady=(10, 6), sticky="w")

        stats = ttk.Frame(f)
        stats.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        stats.columnconfigure(5, weight=1)

        self.lbl_img_cnt = ttk.Label(stats, text="Images:")
        self.lbl_img_cnt.grid(row=0, column=0, padx=(0, 6), sticky="w")
        ttk.Label(stats, textvariable=self.images_count_var, width=8).grid(row=0, column=1, padx=(0, 14), sticky="w")

        self.lbl_lbl_cnt = ttk.Label(stats, text="Labels:")
        self.lbl_lbl_cnt.grid(row=0, column=2, padx=(0, 6), sticky="w")
        ttk.Label(stats, textvariable=self.labels_count_var, width=8).grid(row=0, column=3, padx=(0, 14), sticky="w")

        self.lbl_orph_cnt = ttk.Label(stats, text="Orphans:")
        self.lbl_orph_cnt.grid(row=0, column=4, padx=(0, 6), sticky="w")
        ttk.Label(stats, textvariable=self.orphans_count_var, width=8).grid(row=0, column=5, sticky="w")

        btns = ttk.Frame(f)
        btns.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 10))
        btns.columnconfigure(0, weight=1)

        self.btn_analyze_folders = ttk.Button(btns, command=self.analyze_folders)
        self.btn_analyze_folders.grid(row=0, column=0, sticky="w")

        self.btn_delete_orphans = ttk.Button(btns, command=self.delete_orphan_labels, state="disabled")
        self.btn_delete_orphans.grid(row=0, column=1, padx=8, sticky="w")

    def _build_rename_tab(self):
        f = self.tab_rename
        f.columnconfigure(1, weight=1)

        self.lbl_rename_header = ttk.Label(f, text="Rename", font=("TkDefaultFont", 10, "bold"))
        self.lbl_rename_header.grid(row=0, column=0, columnspan=3, padx=8, pady=(10, 6), sticky="w")

        self.lbl_new_base = ttk.Label(f, text="New name")
        self.lbl_new_base.grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(f, textvariable=self.new_base).grid(row=1, column=1, padx=8, pady=8, sticky="ew")

        self.lbl_start = ttk.Label(f, text="Start")
        self.lbl_start.grid(row=2, column=0, padx=8, pady=8, sticky="w")
        ttk.Spinbox(f, from_=1, to=10_000_000, textvariable=self.start_index, width=10).grid(
            row=2, column=1, padx=8, pady=8, sticky="w"
        )

        self.chk_rename_dry = ttk.Checkbutton(f, variable=self.rename_dry, text="Dry-run")
        self.chk_rename_dry.grid(row=3, column=0, padx=8, pady=8, sticky="w")

        self.btn_rename = ttk.Button(f, command=self.rename_pairs)
        self.btn_rename.grid(row=3, column=2, padx=8, pady=8, sticky="e")

    def _build_classes_tab(self):
        f = self.tab_classes
        f.columnconfigure(1, weight=1)

        self.lbl_check_header = ttk.Label(f, text="Check", font=("TkDefaultFont", 10, "bold"))
        self.lbl_check_header.grid(row=0, column=0, columnspan=3, padx=8, pady=(10, 6), sticky="w")

        self.lbl_expected = ttk.Label(f, text="Expected")
        self.lbl_expected.grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Spinbox(f, from_=1, to=10000, textvariable=self.expected_classes, width=10).grid(
            row=1, column=1, padx=8, pady=6, sticky="w"
        )
        self.btn_analyze = ttk.Button(f, command=self.analyze)
        self.btn_analyze.grid(row=1, column=2, padx=8, pady=6, sticky="e")

        ttk.Separator(f, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=10)

        self.lbl_edit_header = ttk.Label(f, text="Edit", font=("TkDefaultFont", 10, "bold"))
        self.lbl_edit_header.grid(row=3, column=0, columnspan=3, padx=8, pady=(0, 6), sticky="w")

        self.lbl_new_classes = ttk.Label(f, text="New range")
        self.lbl_new_classes.grid(row=4, column=0, padx=8, pady=6, sticky="w")
        ttk.Spinbox(f, from_=1, to=10000, textvariable=self.new_classes, width=10).grid(
            row=4, column=1, padx=8, pady=6, sticky="w"
        )

        self.chk_edit_dry = ttk.Checkbutton(f, variable=self.edit_dry, text="Dry-run")
        self.chk_edit_dry.grid(row=4, column=2, padx=8, pady=6, sticky="e")

        self.box_simple = ttk.LabelFrame(f, text=self.t("simple_reassign"))
        self.box_simple.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 6))
        for c in range(6):
            self.box_simple.columnconfigure(c, weight=0)
        self.box_simple.columnconfigure(5, weight=1)

        self.lbl_old = ttk.Label(self.box_simple, text="Old")
        self.lbl_old.grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Spinbox(self.box_simple, from_=-1, to=10000, textvariable=self.old_class, width=8).grid(
            row=0, column=1, padx=6, pady=6, sticky="w"
        )

        self.lbl_new = ttk.Label(self.box_simple, text="New")
        self.lbl_new.grid(row=0, column=2, padx=6, pady=6, sticky="w")
        ttk.Spinbox(self.box_simple, from_=-1, to=10000, textvariable=self.new_class, width=8).grid(
            row=0, column=3, padx=6, pady=6, sticky="w"
        )

        self.btn_reassign = ttk.Button(self.box_simple, command=self.apply_simple_reassign)
        self.btn_reassign.grid(row=0, column=4, padx=6, pady=6, sticky="e")

        self.box_map = ttk.LabelFrame(f, text=self.t("mapping"))
        self.box_map.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=8, pady=(6, 10))
        self.box_map.columnconfigure(0, weight=1)

        self.mapping_help = ttk.Label(self.box_map, text="Rules", justify="left")
        self.mapping_help.grid(row=0, column=0, padx=6, pady=(6, 2), sticky="w")

        self.mapping_text = tk.Text(self.box_map, height=5, wrap="none")
        self.mapping_text.grid(row=1, column=0, padx=6, pady=6, sticky="ew")

        self.btn_mapping = ttk.Button(self.box_map, command=self.apply_mapping)
        self.btn_mapping.grid(row=2, column=0, padx=6, pady=(0, 6), sticky="e")

    def _apply_i18n(self):
        self.master.title(self.t("title"))

        self.lbl_lang.configure(text=self.t("lang"))
        self.rb_ru.configure(text=self.t("ru"))
        self.rb_en.configure(text=self.t("en"))
        self.btn_log.configure(text=self.t("open_log"))

        nb = self.master.nametowidget(self.master.winfo_children()[0].winfo_children()[1])
        nb.tab(0, text=self.t("tab_folders"))
        nb.tab(1, text=self.t("tab_rename"))
        nb.tab(2, text=self.t("tab_classes"))

        # folders
        self.lbl_images.configure(text=self.t("images_folder"))
        self.lbl_labels.configure(text=self.t("labels_folder"))
        self.btn_images.configure(text=self.t("browse"))
        self.btn_labels.configure(text=self.t("browse"))

        self.lbl_folders_header.configure(text=self.t("folders_header"))
        self.lbl_img_cnt.configure(text=f"{self.t('images_count')}:")
        self.lbl_lbl_cnt.configure(text=f"{self.t('labels_count')}:")
        self.lbl_orph_cnt.configure(text=f"{self.t('orphans_count')}:")
        self.btn_analyze_folders.configure(text=self.t("analyze_folders_btn"))
        self.btn_delete_orphans.configure(text=self.t("delete_orphans_btn"))

        # rename
        self.lbl_rename_header.configure(text=self.t("rename_header"))
        self.lbl_new_base.configure(text=self.t("new_base"))
        self.lbl_start.configure(text=self.t("start_index"))
        self.chk_rename_dry.configure(text=self.t("dry_run"))
        self.btn_rename.configure(text=self.t("rename_btn"))

        # classes
        self.lbl_check_header.configure(text=self.t("check_header"))
        self.lbl_expected.configure(text=self.t("expected_classes"))
        self.btn_analyze.configure(text=self.t("analyze_btn"))

        self.lbl_edit_header.configure(text=self.t("edit_header"))
        self.lbl_new_classes.configure(text=self.t("new_classes"))
        self.chk_edit_dry.configure(text=self.t("edit_dry_run"))

        self.lbl_old.configure(text=self.t("old_class"))
        self.lbl_new.configure(text=self.t("new_class"))
        self.btn_reassign.configure(text=self.t("apply_reassign"))

        if hasattr(self, "box_simple"):
            self.box_simple.configure(text=self.t("simple_reassign"))
        if hasattr(self, "box_map"):
            self.box_map.configure(text=self.t("mapping"))

        self.mapping_help.configure(text=self.t("mapping_rules"))
        self.btn_mapping.configure(text=self.t("apply_mapping"))

        if self.log_window and self.log_window.winfo_exists():
            self.log_window.title(self.t("open_log"))

    def open_log(self):
        if self.log_window and self.log_window.winfo_exists():
            self.log_window.lift()
            return

        w = tk.Toplevel(self.master)
        w.title(self.t("open_log"))
        w.geometry("600x300")
        w.minsize(450, 200)

        txt = tk.Text(w, wrap="word")
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        self.log_window = w
        self.log_text = txt
        self.log("Log opened.")

    def log(self, msg: str):
        if not (self.log_text and self.log_window and self.log_window.winfo_exists()):
            return
        self.log_text.insert("end", msg.rstrip() + "\n")
        self.log_text.see("end")

    def pick_images(self):
        p = filedialog.askdirectory(title=self.t("images_folder"), initialdir=os.getcwd())
        if p:
            self.images_dir.set(p)
            self._maybe_run_folder_analysis()

    def pick_labels(self):
        p = filedialog.askdirectory(title=self.t("labels_folder"), initialdir=os.getcwd())
        if p:
            self.labels_dir.set(p)
            self._maybe_run_folder_analysis()

    def _maybe_run_folder_analysis(self):
        # run analysis only when both folders are chosen
        if self.images_dir.get().strip() and self.labels_dir.get().strip():
            self.analyze_folders()

    def _require_dirs(self) -> tuple[Path, Path] | None:
        img = self.images_dir.get().strip()
        lab = self.labels_dir.get().strip()
        if not img:
            messagebox.showerror(self.t("err"), self.t("need_images"))
            return None
        if not lab:
            messagebox.showerror(self.t("err"), self.t("need_labels"))
            return None
        return Path(img), Path(lab)

    def _image_stems(self, images_folder: Path) -> set[str]:
        stems = set()
        for p in images_folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                stems.add(p.stem)
        return stems

    def _label_stems_and_paths(self, labels_folder: Path) -> tuple[set[str], dict[str, Path]]:
        stems = set()
        mp: dict[str, Path] = {}
        for p in labels_folder.glob("*.txt"):
            stems.add(p.stem)
            mp[p.stem] = p
        return stems, mp

    def analyze_folders(self):
        dirs = self._require_dirs()
        if not dirs:
            return
        images_folder, labels_folder = dirs

        img_stems = self._image_stems(images_folder)
        lbl_stems, lbl_map = self._label_stems_and_paths(labels_folder)

        images_count = len(img_stems)
        labels_count = len(lbl_stems)

        orphan_stems = sorted(lbl_stems - img_stems)
        orphans = [lbl_map[s] for s in orphan_stems if s in lbl_map]

        self.images_count_var.set(str(images_count))
        self.labels_count_var.set(str(labels_count))
        self.orphans_count_var.set(str(len(orphans)))

        self._last_orphan_paths = orphans
        self.btn_delete_orphans.configure(state=("normal" if orphans else "disabled"))

        self.open_log()
        self.log("---- Folder analysis ----")
        self.log(f"Images: {images_count}")
        self.log(f"Labels: {labels_count}")
        self.log(f"Orphan labels (no image): {len(orphans)}")
        if orphans:
            self.log("First 30 orphans:")
            for p in orphans[:30]:
                self.log(f"  {p.name}")
            if len(orphans) > 30:
                self.log("  ...")

            # immediate prompt to delete
            if messagebox.askyesno(self.t("ok"), self.t("delete_orphans_confirm").format(n=len(orphans))):
                self._delete_orphans(orphans)
        else:
            messagebox.showinfo(self.t("ok"), self.t("no_orphans"))

    def delete_orphan_labels(self):
        # manual delete by button
        orphans = list(self._last_orphan_paths)
        if not orphans:
            messagebox.showinfo(self.t("ok"), self.t("no_orphans"))
            return
        if messagebox.askyesno(self.t("ok"), self.t("delete_orphans_confirm").format(n=len(orphans))):
            self._delete_orphans(orphans)

    def _delete_orphans(self, orphans: list[Path]):
        deleted = 0
        self.open_log()
        for p in orphans:
            try:
                p.unlink()
                deleted += 1
                self.log(f"DELETED: {p.name}")
            except Exception as e:
                self.log(f"DELETE ERROR: {p.name} -> {e}")

        # refresh analysis after delete
        self.log(self.t("delete_orphans_done").format(n=deleted))
        messagebox.showinfo(self.t("ok"), self.t("delete_orphans_done").format(n=deleted))
        self.analyze_folders()

    def _index_images(self, images_folder: Path) -> dict[str, Path]:
        idx = {}
        preferred = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        for ext in preferred:
            for p in images_folder.glob(f"*{ext}"):
                idx.setdefault(p.stem, p)
        for p in images_folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                idx.setdefault(p.stem, p)
        return idx

    def rename_pairs(self):
        dirs = self._require_dirs()
        if not dirs:
            return
        images_folder, labels_folder = dirs

        base = self.new_base.get().strip()
        if not base:
            messagebox.showerror(self.t("err"), self.t("need_base"))
            return

        images_index = self._index_images(images_folder)
        label_files = {p.stem: p for p in labels_folder.glob("*.txt")}
        matching = sorted(set(images_index.keys()).intersection(label_files.keys()))
        if not matching:
            messagebox.showinfo(self.t("ok"), "No matching stems.")
            return

        start = int(self.start_index.get())
        dry = bool(self.rename_dry.get())

        self.open_log()
        self.log(f"Rename: matches={len(matching)}, dry={dry}")
        for i, stem in enumerate(matching, start=start):
            img_old = images_index[stem]
            lbl_old = label_files[stem]

            img_new = img_old.with_name(f"{base}_{i}{img_old.suffix}")
            lbl_new = lbl_old.with_name(f"{base}_{i}{lbl_old.suffix}")

            self.log(f"[{i}] {img_old.name} -> {img_new.name}")
            self.log(f"[{i}] {lbl_old.name} -> {lbl_new.name}")

            if dry:
                continue

            if img_new.exists() or lbl_new.exists():
                self.log("  SKIP: target exists")
                continue

            try:
                img_old.rename(img_new)
                lbl_old.rename(lbl_new)
            except Exception as e:
                self.log(f"  ERROR: {e}")

        if dry:
            messagebox.showinfo(self.t("ok"), self.t("dry_note"))
        else:
            messagebox.showinfo(self.t("ok"), self.t("rename_done"))

    def _parse_yolo_line(self, line: str) -> tuple[int, list[str]] | None:
        s = line.strip()
        if not s:
            return None
        parts = re.split(r"\s+", s)
        if len(parts) < 5:
            return None
        try:
            cls = int(parts[0])
            return cls, parts
        except Exception:
            return None

    def _txt_files(self, labels_folder: Path) -> list[Path] | None:
        txt_files = sorted(labels_folder.glob("*.txt"))
        if not txt_files:
            messagebox.showerror(self.t("err"), self.t("no_txt"))
            return None
        return txt_files

    def analyze(self):
        dirs = self._require_dirs()
        if not dirs:
            return
        _, labels_folder = dirs

        txt_files = self._txt_files(labels_folder)
        if txt_files is None:
            return

        expected = int(self.expected_classes.get())
        dataset_max = None
        unique = set()
        out_of_range = 0
        bad_lines = 0

        self.open_log()
        self.log(f"Analyze: expected={expected}, files={len(txt_files)}")

        for p in txt_files:
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception as e:
                self.log(f"READ ERROR: {p.name} -> {e}")
                continue

            for line in lines:
                if not line.strip():
                    continue
                parsed = self._parse_yolo_line(line)
                if parsed is None:
                    bad_lines += 1
                    continue
                cls, _ = parsed
                unique.add(cls)
                dataset_max = cls if dataset_max is None else max(dataset_max, cls)
                if cls < 0 or cls >= expected:
                    out_of_range += 1

        inferred = (dataset_max + 1) if dataset_max is not None else 0
        self.log(f"Dataset max class_id: {dataset_max}")
        self.log(f"Inferred class count (max+1): {inferred}")
        self.log(f"Unique class_id: {sorted(unique)[:200]}{' ...' if len(unique) > 200 else ''}")
        self.log(f"Out of range count: {out_of_range}")
        self.log(f"Bad line count: {bad_lines}")

        messagebox.showinfo(self.t("ok"), self.t("analyze_done"))

    def _apply_edit(self, transform_fn):
        dirs = self._require_dirs()
        if not dirs:
            return
        _, labels_folder = dirs

        txt_files = self._txt_files(labels_folder)
        if txt_files is None:
            return

        new_classes = int(self.new_classes.get())
        dry = bool(self.edit_dry.get())

        self.open_log()
        self.log(f"Edit: files={len(txt_files)}, new_classes={new_classes}, dry={dry}")

        planned = []
        changed_files = 0
        changed_lines_total = 0

        for p in txt_files:
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception as e:
                self.log(f"READ ERROR: {p.name} -> {e}")
                return

            new_lines = []
            changed_lines = 0

            for ln, line in enumerate(lines, start=1):
                if not line.strip():
                    new_lines.append(line)
                    continue

                parsed = self._parse_yolo_line(line)
                if parsed is None:
                    new_lines.append(line)
                    continue

                cls, parts = parsed
                new_cls = transform_fn(cls)

                if new_cls is None:
                    changed_lines += 1
                    continue

                if new_cls < 0 or new_cls >= new_classes:
                    messagebox.showerror(self.t("err"),
                                         f"{p.name}:{ln} class_id {cls} -> {new_cls} out of range 0..{new_classes-1}")
                    return

                if new_cls != cls:
                    parts[0] = str(new_cls)
                    changed_lines += 1

                new_lines.append(" ".join(parts))

            new_text = "\n".join(new_lines) + ("\n" if lines else "")
            planned.append((p, new_text, changed_lines))

        for p, text, changed_lines in planned:
            if changed_lines > 0:
                changed_files += 1
                changed_lines_total += changed_lines

            if dry:
                continue

            try:
                p.write_text(text, encoding="utf-8")
            except Exception as e:
                self.log(f"WRITE ERROR: {p.name} -> {e}")
                messagebox.showerror(self.t("err"), f"Write error: {p.name}\n{e}")
                return

        self.log(f"Changed files: {changed_files}/{len(planned)}")
        self.log(f"Changed lines total: {changed_lines_total}")

        if dry:
            messagebox.showinfo(self.t("ok"), self.t("dry_note"))
        else:
            messagebox.showinfo(self.t("ok"), self.t("edit_done"))

    def apply_simple_reassign(self):
        old_id = int(self.old_class.get())
        new_id = int(self.new_class.get())

        def transform(cls: int):
            return new_id if cls == old_id else cls

        self._apply_edit(transform)

    def _parse_mapping_rules(self) -> dict[int, int] | None:
        raw = self.mapping_text.get("1.0", "end").splitlines()
        mapping: dict[int, int] = {}

        for i, line in enumerate(raw, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"^\s*(-?\d+)\s*(?:->|:)\s*(-?\d+)\s*$", s)
            if not m:
                messagebox.showerror(self.t("err"), f"Bad rule at line {i}: {line}")
                return None
            mapping[int(m.group(1))] = int(m.group(2))

        if not mapping:
            messagebox.showerror(self.t("err"), "No mapping rules.")
            return None
        return mapping

    def apply_mapping(self):
        mapping = self._parse_mapping_rules()
        if mapping is None:
            return

        def transform(cls: int):
            return mapping.get(cls, cls)

        self._apply_edit(transform)


def main():
    root = tk.Tk()
    root.geometry("670x520")
    root.minsize(580, 420)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
