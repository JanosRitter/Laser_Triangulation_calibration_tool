import csv
from pathlib import Path


def _convert_csv_value(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "":
        return ""

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "nan":
        return float("nan")

    try:
        if "." in stripped or "e" in lowered:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def save_fit_table_csv(fit_table: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(fit_table) == 0:
        raise ValueError("fit_table ist leer.")

    fieldnames = list(fit_table[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fit_table)


def load_fit_table_csv(input_path: str | Path) -> list[dict]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fit-Tabelle nicht gefunden: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _convert_csv_value(value) for key, value in row.items()}
            for row in reader
        ]
