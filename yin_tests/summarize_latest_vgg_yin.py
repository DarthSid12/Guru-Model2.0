"""Collect the final-checkpoint Yin results for the two latest VGG2k models."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/yin_latest_vgg_20260927"
NOAA_CROSS = ROOT / "runs/yin_noaa_two_cross_20260927/results"
MODELS = ("r21vgg2k_vgg16_bn_s42", "r21vgg2k_vgg16_bn_aa5_s42")
FACES = ("faces_rfwWM64", "faces_cfdWM64")
CATEGORIES = ("face", "objects", "houses_yin64")
CONDITIONS = ("UU", "II", "UI", "IU")


def main():
    rows = []
    for model in MODELS:
        for face in FACES:
            cross = (NOAA_CROSS / face if model == MODELS[0] else
                     OUT / "two_cross" / model / face)
            for protocol in ("two_cross", "single_noise"):
                categories = CATEGORIES
                for category in categories:
                    folder = (cross if protocol == "two_cross" and category == "face" else
                              OUT / "transfer" / "results" / model / face /
                              category if protocol == "two_cross" else
                              OUT / protocol / model / face /
                              (face if category == "face" else category))
                    done = folder / "DONE.json"
                    summary = folder / "summary.json"
                    if not done.exists() or not summary.exists():
                        raise RuntimeError(f"Incomplete: {folder}")
                    data = json.loads(summary.read_text())
                    expected_rows = 250 if protocol == "two_cross" and category != "face" else 200
                    if json.loads(done.read_text())["rows"] != expected_rows:
                        raise RuntimeError(f"Expected {expected_rows} seed rows: {folder}")
                    for condition in CONDITIONS:
                        key = "UI_cross" if protocol == "two_cross" and category != "face" and condition == "UI" else condition
                        value = data["cells" if protocol == "two_cross" and category != "face" else "conditions"][key]
                        if value["n"] != 50:
                            raise RuntimeError(f"Expected 50 seeds: {folder} {condition}")
                        rows.append(dict(model=model, protocol=protocol,
                                         calibration_face=face,
                                         category=face if category == "face" else category,
                                         condition=condition,
                                         mean_pct=value["mean_pct"],
                                         sd_pct=value["sd_pct"], n_seeds=value["n"],
                                         source=str(folder.relative_to(ROOT))))
    path = OUT / "all_conditions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Latest VGG2k final-checkpoint Yin results", "",
              "50 seeds (101–150). Each face calibration is independent.",
              "Single-noise uses one UU-fitted p for all four orientations and",
              "transfers it to faces, objects, and houses. Two-cross fits p1 on",
              "UU and p2 on II, then applies p1/p2 to UI and p2/p1 to IU.", "",
              "| Model | Protocol | Calibration | Category | UU | II | UI | IU |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for protocol in ("two_cross", "single_noise"):
            for face in FACES:
                for category in (face, "objects", "houses_yin64"):
                    cells = {r["condition"]: r for r in rows
                             if (r["model"], r["protocol"], r["calibration_face"],
                                 r["category"]) == (model, protocol, face, category)}
                    values = [f"{cells[c]['mean_pct']:.2f} ± {cells[c]['sd_pct']:.2f}"
                              for c in CONDITIONS]
                    report.append("| " + " | ".join((model, protocol, face,
                                                     category, *values)) + " |")
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    (OUT / "STATUS").write_text("complete\n")
    print(f"Aggregated {len(rows)} conditions")


if __name__ == "__main__":
    main()
