"""Summarize completed 200-seed no-AA VGG16 Yin runs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/yin_noaa_200_20260928"
FACES = ("faces_rfwWM64", "faces_cfdWM64")
SETUPS = {
    "single_uu": ("UU", "II_single", "UI_shared", "IU_single"),
    "two_cross": ("UU", "II_cross", "UI_cross", "IU_cross"),
    "two_shared_ui": ("UU", "II_cross", "UI_shared", "IU_cross"),
}


def main():
    records = []
    trials = []
    for face in FACES:
        for category in (face, "objects"):
            folder = OUT / "results" / face / category
            if not (folder / "DONE.json").exists():
                raise RuntimeError(f"Incomplete: {folder}")
            data = json.loads((folder / "summary.json").read_text())
            with (folder / "results.csv").open(newline="") as handle:
                rows = {(int(r["seed"]), r["cell"]): r for r in csv.DictReader(handle)}
            if len(rows) != 1400:
                raise RuntimeError(f"Expected 1400 distinct trials: {folder}")
            for setup, cells in SETUPS.items():
                for condition, cell in zip(("UU", "II", "UI", "IU"), cells):
                    stat = data["cells"][cell]
                    if stat["n"] != 200:
                        raise RuntimeError(f"Expected 200 seeds: {folder} {cell}")
                    records.append(dict(model=data["model"], calibration_face=face,
                                        category=category, setup=setup, condition=condition,
                                        p1=data["p1"], p2=data["p2"],
                                        mean_pct=stat["mean_pct"], sd_pct=stat["sd_pct"],
                                        n_seeds=200, calibration_seeds=50,
                                        source_csv=str((folder / "results.csv").relative_to(ROOT)),
                                        source_cell=cell))
                    for seed in range(101, 301):
                        row = rows[(seed, cell)]
                        trials.append(dict(calibration_face=face, category=category,
                                           setup=setup, condition=condition, seed=seed,
                                           p_study=row["p_study"], p_test=row["p_test"],
                                           accuracy_pct=row["accuracy_pct"],
                                           source_kind=row["source_kind"], source_cell=cell))
    for name, rows in (("all_conditions.csv", records), ("all_trials.csv", trials)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = ["# No-AA VGG16 Yin: 200-seed results", "",
              "Final epoch-124 checkpoint `r21vgg2k_vgg16_bn_s42`.",
              "Noise rates were fitted on seeds 101–150 and held fixed for evaluation",
              "on seeds 101–300. The first 50 trial seeds are reused from the completed",
              "runs. Each calibration face is independent. Values are mean ± SD (%).", "",
              "| Setup | UU | II | UI | IU |",
              "| --- | --- | --- | --- | --- |",
              "| single_uu | p1/p1 | p1/p1 | p1/p1 | p1/p1 |",
              "| two_cross | p1/p1 | p2/p2 | p1/p2 | p2/p1 |",
              "| two_shared_ui | p1/p1 | p2/p2 | p1/p1 | p2/p1 |", "",
              "Each pair is study/test noise. Noise is applied at both phases and",
              "to both old/new test alternatives. Face anchors are RFW WM64 and",
              "CFD WM64; evaluations include the corresponding human faces and objects.", "",
              "| Calibration | Category | Setup | p1 | p2 | UU | II | UI | IU |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for face in FACES:
        for category in (face, "objects"):
            for setup in SETUPS:
                cells = [next(r for r in records if (r["calibration_face"], r["category"],
                    r["setup"], r["condition"]) == (face, category, setup, condition))
                    for condition in ("UU", "II", "UI", "IU")]
                values = [f"{r['mean_pct']:.2f} ± {r['sd_pct']:.2f}" for r in cells]
                report.append("| " + " | ".join((face, category, setup,
                    f"{cells[0]['p1']:.2f}", f"{cells[0]['p2']:.2f}", *values)) + " |")
    report += ["", "`all_trials.csv` contains every setup's per-seed rows;",
               "`results/` contains the seven distinct evaluated cells and provenance.", ""]
    (OUT / "REPORT.md").write_text("\n".join(report))
    (OUT / "STATUS").write_text("complete\n")
    print(f"Aggregated {len(records)} conditions and {len(trials)} trial rows")


if __name__ == "__main__":
    main()
