"""Check Git-visible files for patient-level exports and saved notebook outputs.

This is a structural safeguard, not a proof of complete clinical anonymity.
Run before staging files; it also examines already tracked files, which .gitignore
cannot protect retroactively.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CSV = {
    "01_словарь.csv", "01_исключённые_столбцы.csv", "02_сравнение_моделей.csv",
    "02_метрики_блоков.csv", "02_метрики_повторений.csv", "02_условные_bootstrap_CI.csv",
    "02_условные_сравнения.csv", "02_условные_метрики_блоков.csv",
    "02_условные_метрики_повторений.csv", "02_парные_изменения.csv",
    "02_парные_изменения_блоков.csv", "02_проверка_состава_признаков.csv",
    "03_mapping_признаков.csv", "03_важности_SHAP.csv",
    "03_важности_преобразованных_признаков.csv", "03_средние_метрики_внешних_блоков.csv",
    "03_условные_средние_блоков.csv", "03_эффект_Optuna.csv",
    "03_ТАГ_МС_разности_средних_блоков.csv", "сравнение_SHAP_до_после.csv",
    "сравнение_до_после_исключения_индикатора_СРБ.csv",
    "проверка_внешних_численностей_нового_запуска.csv"
}
ROW_FIELDS = {"row_id", "group_id", "y_true", "p_death", "position", "background_row_id",
              "patient_id", "record_id", "display_data", "value_counts"}
FORBIDDEN_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".pdf", ".docx", ".doc", ".joblib",
                      ".pkl", ".pickle", ".npz", ".npy", ".sqlite", ".sqlite3", ".db", ".tsv"}


def scan(root=ROOT, *, use_git=True):
    if use_git:
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=root, capture_output=True, check=True)
        candidates = [root / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    else:
        skipped = {".git", "private", ".idea", ".venv", "__pycache__", ".ipynb_checkpoints"}
        candidates = [p for p in root.rglob("*") if p.is_file() and not skipped.intersection(p.relative_to(root).parts)]
    errors = []
    for path in candidates:
        if not path.exists():
            continue
        relative = path.relative_to(root)
        if "private" in relative.parts:
            errors.append(str(relative) + ": private clinical artifacts are Git-visible")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(str(relative) + ": restricted file format")
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text())
            for index, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                    errors.append(f"{relative}: cell {index} contains saved outputs/execution history")
                if cell.get("attachments"):
                    errors.append(f"{relative}: embedded notebook attachment needs review")
        if path.suffix == ".csv":
            if path.name not in ALLOWED_CSV or "results" not in relative.parts:
                errors.append(str(relative) + ": CSV is not an approved aggregate")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle), [])
            forbidden = set(columns).intersection(ROW_FIELDS)
            direct = [c for c in columns if re.search(r"^(фио|фамилия|имя|отчество|телефон|адрес|дата рождения|номер истории болезни)$", c.strip(), re.I)]
            if forbidden or direct:
                errors.append(str(relative) + ": patient-level/identifier fields")
        if path.suffix == ".json":
            obj = json.loads(path.read_text())
            def inspect(value):
                if isinstance(value, dict):
                    if ROW_FIELDS.intersection(value):
                        errors.append(str(relative) + ": JSON contains patient-level fields")
                    for item in value.values(): inspect(item)
                elif isinstance(value, list):
                    for item in value: inspect(item)
            inspect(obj)
        if path.suffix in {".md", ".py", ".json", ".ipynb", ".txt", ".toml"}:
            text = path.read_text(encoding="utf-8")
            prefixes = ["/" + "Users" + "/", "/" + "home" + "/"]
            absolute_path = "(?:" + "|".join(map(re.escape, prefixes)) + r")[^\s\"']+"
            if re.search(absolute_path, text) or ("file:" + "///") in text:
                errors.append(str(relative) + ": machine-specific absolute path")
    return len(candidates), sorted(set(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-git", action="store_true", help="Check export staging tree before it is copied into a Git checkout")
    args = parser.parse_args()
    count, errors = scan(use_git=not args.no_git)
    print(json.dumps({"files_checked": count, "status": "failed" if errors else "passed", "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
