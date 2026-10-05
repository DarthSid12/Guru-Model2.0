"""Aggregate RFW/CFD two-cross Yin results for the final no-AA VGG2k."""
import csv
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/yin_noaa_two_cross_20260927"
FACES = ("faces_rfwWM64", "faces_cfdWM64")
CONDITIONS = ("UU", "II", "UI", "IU")


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buffer.getvalue())
    os.replace(tmp, path)


def aggregate():
    cells = []
    trials = []
    table = []
    complete = 0
    for face in FACES:
        folder = OUT / "results" / face
        summary_path = folder / "summary.json"
        result_path = folder / "results.csv"
        label = "RFW WM64" if "rfw" in face else "CFD WM64"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() and (folder / "DONE.json").exists() else None
        seen = {}
        if result_path.exists():
            with result_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = int(row["seed"]), row["study"] + row["test"]
                    if key in seen:
                        raise ValueError(f"Duplicate result row: {result_path}, {key}")
                    seen[key] = row
        for condition in CONDITIONS:
            stat = summary["conditions"][condition] if summary else None
            cells.append(dict(model="r21vgg2k_vgg16_bn_s42", face_set=face,
                setup="two_cross", condition=condition,
                p1=summary["calibration"]["U"]["p"] if summary else "",
                p2=summary["calibration"]["I"]["p"] if summary else "",
                mean_pct=stat["mean_pct"] if stat else "",
                sd_pct=stat["sd_pct"] if stat else "", n_seeds=stat["n"] if stat else 0,
                status="done" if stat else "pending",
                source_csv=str(result_path.relative_to(ROOT))))
            complete += stat is not None
            for seed in range(101, 151):
                if (seed, condition) in seen:
                    row = seen[(seed, condition)]
                    trials.append(dict(face_set=face, setup="two_cross",
                        condition=condition, seed=seed,
                        p_study=row["p_study"], p_test=row["p_test"],
                        accuracy_pct=row["accuracy_pct"],
                        source_csv=str(result_path.relative_to(ROOT))))
        if summary:
            fit_u, fit_i = summary["calibration"]["U"], summary["calibration"]["I"]
            stats = [f"{summary['conditions'][c]['mean_pct']:.2f} ± {summary['conditions'][c]['sd_pct']:.2f}"
                     for c in CONDITIONS]
            table.append(f"| {label} | {fit_u['p']:.2f} | {fit_i['p']:.2f} | " +
                         " | ".join(stats) +
                         f" | {fit_u['residual_pp']:+.2f} | {fit_i['residual_pp']:+.2f} |")
        else:
            table.append(f"| {label} | pending | pending | pending | pending | pending | pending | pending | pending |")
    write_csv(OUT / "all_conditions.csv", cells)
    write_csv(OUT / "all_trials.csv", trials)
    report = ["# No-AA VGG2k: two-cross Yin on faces", "",
        "Final epoch-124 checkpoint. RFW and CFD white-male face sets each fit",
        "p1 on UU toward 96.29% and p2 on II toward 81.88%, independently",
        "over seeds 101–150. Noise is applied to the sampled 256-bit code",
        "at study and at perception for both old/new test alternatives.", "",
        "| Condition | UU | II | UI | IU |",
        "| --- | --- | --- | --- | --- |",
        "| Study/test p | p1/p1 | p2/p2 | p1/p2 | p2/p1 |", "",
        "Forty studied items, 24 new items, 10 study fixations, 32 test",
        "fixations, KDE width 2, and 24 old/new pairs per seed. Faces are",
        "reshuffled by seed. Values below are mean ± SD (%).", "",
        f"Completed face sets: **{complete // 4}/2**.", "",
        "| Face set | p1 | p2 | UU | II | UI | IU | UU − human (pp) | II − human (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Human reference | — | — | 96.29 | 81.88 | 84.13 | 78.58 | — | — |",
        *table, "",
        "Calibration curves and fit status are in each face set's",
        "`calibration.json`; per-seed accuracy is in `all_trials.csv`.", ""]
    (OUT / "REPORT.md").write_text("\n".join(report))
    print(f"Aggregated {complete}/8 cells", flush=True)


if __name__ == "__main__":
    aggregate()
