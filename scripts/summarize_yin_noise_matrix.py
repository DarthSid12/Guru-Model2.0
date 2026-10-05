"""Combine the completed Yin experiments into one auditable 3x2x2x3 matrix."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "runs/yin_single_noise_20260925/results"
TWO = ROOT / "runs/yin_orientation_20260925/results/final"
MODELS = ("r21dev_vgg16_bn_aa_s42", "r21vgg2k_vgg16_bn_aa_s42")
FACES = ("faces_rfwWM64", "faces_cfdWM64")
CONDITIONS = ("UU", "II", "UI", "IU")
SETS = ("faces", "objects", "houses_yin64")
HUMAN = {"faces": (96.29, 81.88, 84.13, 78.58),
         "objects": (84.79, 83.96, 86.71, 82.75),
         "houses_yin64": (90.71, 85.75, 88.08, 85.71)}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buffer.getvalue())
    os.replace(tmp, path)


def source_map(model, face, category, setup, out):
    one_path = SINGLE / model / face / category / "results.csv"
    two_path = TWO / model / face / "results.csv"
    new_path = out / "results" / model / face / category / "results.csv"
    if setup == "single_uu":
        return {condition: (one_path, condition, "condition") for condition in CONDITIONS}
    if category == face:
        mapping = {condition: (two_path, condition, "study_test") for condition in CONDITIONS}
        if setup == "two_shared_ui":
            mapping["UI"] = one_path, "UI", "condition"
        return mapping
    mapping = {"UU": (one_path, "UU", "condition"), "II": (new_path, "II", "cell"),
               "UI": (new_path, "UI_cross", "cell"), "IU": (new_path, "IU", "cell")}
    if setup == "two_shared_ui":
        mapping["UI"] = one_path, "UI", "condition"
    return mapping


def indexed(path, kind, cache):
    key = str(path), kind
    if key not in cache:
        rows = read_csv(path)
        column = {"condition": lambda r: r["condition"],
                  "study_test": lambda r: r["study"] + r["test"],
                  "cell": lambda r: r["cell"]}[kind]
        indexed_rows = {(int(row["seed"]), column(row)): row for row in rows}
        if len(indexed_rows) != len(rows):
            raise ValueError(f"Duplicate result rows: {path}")
        cache[key] = indexed_rows
    return cache[key]


def aggregate(out):
    conditions = []
    trials = []
    calibrations = []
    cache = {}
    finished = 0
    for model in MODELS:
        for face in FACES:
            one_fit = json.loads((SINGLE / model / face / "calibration.json").read_text())
            two_fit = json.loads((TWO / model / face / "calibration.json").read_text())
            p1, p2 = float(two_fit["U"]["p"]), float(two_fit["I"]["p"])
            if p1 != one_fit["p"]:
                raise ValueError(f"Calibrations disagree: {model}/{face}")
            calibrations.append(dict(model=model, calibration_face=face,
                epoch=124, p1=p1, uu_target_pct=100 * two_fit["U"]["target"],
                uu_achieved_pct=100 * two_fit["U"]["achieved"],
                uu_residual_pp=two_fit["U"]["residual_pp"],
                p1_fit_status=two_fit["U"]["status"], p2=p2,
                ii_target_pct=100 * two_fit["I"]["target"],
                ii_achieved_pct=100 * two_fit["I"]["achieved"],
                ii_residual_pp=two_fit["I"]["residual_pp"],
                p2_fit_status=two_fit["I"]["status"]))
            for dataset in SETS:
                category = face if dataset == "faces" else dataset
                for setup in ("single_uu", "two_cross", "two_shared_ui"):
                    mapping = source_map(model, face, category, setup, out)
                    for condition in CONDITIONS:
                        path, key, kind = mapping[condition]
                        result = dict(model=model, calibration_face=face,
                            dataset=dataset, setup=setup, condition=condition,
                            p1=p1, p2=("" if setup == "single_uu" else p2),
                            mean_pct="", sd_pct="", n_seeds=0,
                            source_csv=str(path.relative_to(ROOT)),
                            source_cell=key, status="queued")
                        if path.exists():
                            rows = indexed(path, kind, cache)
                            values = []
                            for seed in range(101, 151):
                                row = rows.get((seed, key))
                                if row is None:
                                    continue
                                accuracy = float(row["accuracy_pct"])
                                values.append(accuracy)
                                trials.append(dict(model=model, calibration_face=face,
                                    dataset=dataset, setup=setup, condition=condition,
                                    seed=seed, accuracy_pct=accuracy,
                                    source_csv=str(path.relative_to(ROOT)), source_cell=key))
                            if values:
                                result.update(mean_pct=statistics.mean(values),
                                    sd_pct=statistics.stdev(values) if len(values) > 1 else 0.0,
                                    n_seeds=len(values), status="done" if len(values) == 50 else "partial")
                        finished += result["status"] == "done"
                        conditions.append(result)
    write_csv(out / "all_conditions.csv", list(conditions[0]), conditions)
    write_csv(out / "all_trials.csv", list(trials[0]), trials)
    write_csv(out / "calibrations.csv", list(calibrations[0]), calibrations)
    human_rows = [dict(dataset=dataset, **dict(zip(CONDITIONS, values)))
                  for dataset, values in HUMAN.items()]
    write_csv(out / "human_references.csv", list(human_rows[0]), human_rows)
    report = ["# Yin noise matrix", "",
        "| Setup | UU | II | UI | IU |",
        "| --- | --- | --- | --- | --- |",
        "| `single_uu` | p1/p1 | p1/p1 | p1/p1 | p1/p1 |",
        "| `two_cross` | p1/p1 | p2/p2 | p1/p2 | p2/p1 |",
        "| `two_shared_ui` | p1/p1 | p2/p2 | p1/p1 | p2/p1 |", "",
        "Each pair is study/test noise. p1 is fitted on face UU; p2 is fitted independently on face II.",
        "Both study and test encodings receive their listed noise.", "",
        "Final epoch 124; 50 seeds (101–150); accuracy is percent correct in 24 old/new pairs per seed.",
        "The human face row is Yin's reference, not measurements on RFW or CFD images.", "",
        f"Completed cells: **{finished}/{len(conditions)}**. `all_trials.csv` contains the per-seed data.", "",
        "| Model | Face calibration | Dataset | Setup | p1 | p2 | UU | II | UI | IU |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, vals in HUMAN.items():
        label = "houses" if dataset == "houses_yin64" else dataset
        report.append("| Human reference | — | " + label + " | — | — | — | " +
                      " | ".join(f"{value:.2f}" for value in vals) + " |")
    by_group = {}
    for row in conditions:
        group = row["model"], row["calibration_face"], row["dataset"], row["setup"]
        by_group.setdefault(group, {})[row["condition"]] = row
    for (model, face, dataset, setup), rows in by_group.items():
        label = "Dev VGG" if model.startswith("r21dev") else "VGG2k"
        face_label = "RFW WM64" if "rfw" in face else "CFD WM64"
        cells = []
        for condition in CONDITIONS:
            row = rows[condition]
            cells.append((f"{row['mean_pct']:.2f} ± {row['sd_pct']:.2f}"
                          if row["status"] == "done" else "pending"))
        report.append(f"| {label} | {face_label} | {dataset} | {setup} | " +
                      f"{rows['UU']['p1']:.2f} | " +
                      ("—" if setup == "single_uu" else f"{rows['UU']['p2']:.2f}") +
                      " | " + " | ".join(cells) + " |")
    category_out = ROOT / "runs/yin_category_uu_20260926"
    report += ["", "## Category-specific UU calibration", "",
        "One p per model and dataset is fitted to that dataset's human UU value:",
        "objects 84.79% or houses 90.71%. The same p applies at study and test",
        "in all four conditions. These rows use the final epoch-124 checkpoints",
        "and the same 50 seeds as the table above. If the target is unreachable,",
        "the nearest sampled p and its residual are reported in the detailed",
        "category results. Values are mean ± SD (%).", "",
        "| Model | Dataset | p | UU | II | UI | IU | UU − human (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in ("objects", "houses_yin64"):
        label = "Houses" if dataset == "houses_yin64" else "Objects"
        report.append(f"| Human | {label} | — | " +
                      " | ".join(f"{v:.2f}" for v in HUMAN[dataset]) + " | — |")
    for model in MODELS:
        for dataset in ("objects", "houses_yin64"):
            folder = category_out / "results" / model / dataset
            summary_path = folder / "summary.json"
            label = "Dev VGG" if model.startswith("r21dev") else "VGG2k"
            dataset_label = "Houses" if dataset == "houses_yin64" else "Objects"
            if summary_path.exists() and (folder / "DONE.json").exists():
                summary = json.loads(summary_path.read_text())
                p = f"{summary['p_noise']:.2f}"
                vals = [f"{summary['conditions'][c]['mean_pct']:.2f} ± {summary['conditions'][c]['sd_pct']:.2f}"
                        for c in CONDITIONS]
                residual = f"{summary['residual_pp']:+.2f}"
            else:
                p = "pending"
                vals = ["pending"] * 4
                residual = "pending"
            report.append(f"| {label} | {dataset_label} | {p} | " + " | ".join(vals) + f" | {residual} |")
    report += ["", "Full category-calibrated results: [report](../yin_category_uu_20260926/REPORT.md).", ""]
    report += ["", "`all_conditions.csv` gives each cell's exact mean, SD, sample count,",
               "and source file. `all_trials.csv` gives every seed's accuracy and source.",
               "`calibrations.csv` records fitted rates and residuals; `human_references.csv`",
               "records the human values.", ""]
    (out / "REPORT.md").write_text("\n".join(report))
    return finished, len(conditions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    done, total = aggregate(args.out_root)
    print(f"Aggregated {done}/{total} cells", flush=True)
