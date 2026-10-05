"""Summarize category-specific UU calibrations for two final VGG checkpoints."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import csv
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/yin_category_uu_20260926"
MODELS = ("r21dev_vgg16_bn_aa_s42", "r21vgg2k_vgg16_bn_aa_s42")
DATASETS = ("objects", "houses_yin64")
HUMAN = {"objects": (84.79, 83.96, 86.71, 82.75),
         "houses_yin64": (90.71, 85.75, 88.08, 85.71)}
CONDITIONS = ("UU", "II", "UI", "IU")


def write_csv(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buffer.getvalue())
    os.replace(tmp, path)


def table_rows():
    rows = []
    for category, human in HUMAN.items():
        rows.append(f"| Human | {category} | — | " + " | ".join(f"{v:.2f}" for v in human) + " | — |")
    for model in MODELS:
        for category in DATASETS:
            folder = OUT / "results" / model / category
            summary_path = folder / "summary.json"
            label = "Dev VGG" if model.startswith("r21dev") else "VGG2k"
            if summary_path.exists() and (folder / "DONE.json").exists():
                summary = json.loads(summary_path.read_text())
                values = [f"{summary['conditions'][c]['mean_pct']:.2f} ± {summary['conditions'][c]['sd_pct']:.2f}"
                          for c in CONDITIONS]
                p = f"{summary['p_noise']:.2f}"
                residual = f"{summary['residual_pp']:+.2f}"
            else:
                values = ["pending"] * 4
                p = "pending"
                residual = "pending"
            rows.append(f"| {label} | {category} | {p} | " + " | ".join(values) + f" | {residual} |")
    return rows


def aggregate():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    trials = []
    complete = 0
    for model in MODELS:
        for category in DATASETS:
            folder = OUT / "results" / model / category
            summary_path = folder / "summary.json"
            result_path = folder / "results.csv"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() and (folder / "DONE.json").exists() else None
            seen = {}
            if result_path.exists():
                with result_path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        key = int(row["seed"]), row["condition"]
                        if key in seen:
                            raise ValueError(f"Duplicate result row: {result_path}, {key}")
                        seen[key] = row
            for condition in CONDITIONS:
                source = str(result_path.relative_to(ROOT))
                stat = summary["conditions"][condition] if summary else None
                cells.append(dict(model=model, dataset=category, condition=condition,
                    target_uu_pct=100 * (0.8479 if category == "objects" else 0.9071),
                    p_noise=summary["p_noise"] if summary else "",
                    mean_pct=stat["mean_pct"] if stat else "",
                    sd_pct=stat["sd_pct"] if stat else "",
                    n_seeds=stat["n"] if stat else 0,
                    status="done" if stat else "pending", source_csv=source))
                complete += stat is not None
                for seed in range(101, 151):
                    if (seed, condition) in seen:
                        row = seen[(seed, condition)]
                        trials.append(dict(model=model, dataset=category,
                            condition=condition, seed=seed,
                            accuracy_pct=row["accuracy_pct"], p_noise=row["p_study"],
                            source_csv=source))
    write_csv(OUT / "all_conditions.csv", cells)
    if trials:
        write_csv(OUT / "all_trials.csv", trials)
    report = ["# Category-calibrated Yin: objects and houses", "",
        "One noise probability is fitted separately for each model and dataset",
        "on human UU (objects 84.79%; houses 90.71%). The same probability is",
        "used at study and test for UU, II, UI, and IU. Final epoch 124, 50",
        "seeds (101–150), 24 old/new pairs per seed. If a target is unreachable,",
        "the closest sampled p and signed residual are recorded. Values are mean ± SD (%).", "",
        f"Completed model rows: **{complete // 4}/4**.", "",
        "| Model | Dataset | p | UU | II | UI | IU | UU − human (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *table_rows(), "",
        "`all_conditions.csv` contains exact statistics and `all_trials.csv`",
        "contains available per-seed results. Calibration curves and samples",
        "are in each model/dataset result directory.", ""]
    (OUT / "REPORT.md").write_text("\n".join(report))
    print(f"Aggregated {complete}/16 cells", flush=True)


if __name__ == "__main__":
    aggregate()
