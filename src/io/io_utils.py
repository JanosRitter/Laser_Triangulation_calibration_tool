import csv
from pathlib import Path


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