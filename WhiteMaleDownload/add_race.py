#!/usr/bin/env python3

import argparse
import csv
import xml.etree.ElementTree as ET


ETHNICITY_NAMES = {
    1: "African American",
    2: "East Asian",
    3: "Caucasian Latin",
    4: "Asian Indian",
}


def read_ethnicity_xml(xml_file):
    """Read VMER ethnicity annotations from an XML file."""

    tree = ET.parse(xml_file)
    root = tree.getroot()

    ethnicity_map = {}

    for subject in root.findall(".//subject"):
        id_element = subject.find("id")
        ethnicity_element = subject.find("ethnicity")

        if id_element is None or ethnicity_element is None:
            continue

        class_id = id_element.text.strip()
        ethnicity = int(ethnicity_element.text.strip())

        ethnicity_map[class_id] = ethnicity

    return ethnicity_map


def main():

    parser = argparse.ArgumentParser(
        description="Add VMER ethnicity labels to VGGFace2 identity_meta.csv"
    )

    parser.add_argument(
        "--identity",
        default="identity_meta.csv",
        help="VGGFace2 identity metadata CSV"
    )

    parser.add_argument(
        "--train",
        default="finalTrain.xml",
        help="VMER finalTrain.xml"
    )

    parser.add_argument(
        "--test",
        default="finalTest.xml",
        help="VMER finalTest.xml"
    )

    parser.add_argument(
        "--output",
        default="identity_meta_race.csv",
        help="Output CSV"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Read ethnicity annotations
    # ------------------------------------------------------------

    train_race = read_ethnicity_xml(args.train)
    test_race = read_ethnicity_xml(args.test)

    race_map = {}

    race_map.update(train_race)
    race_map.update(test_race)

    print(f"Train annotations: {len(train_race):,}")
    print(f"Test annotations:  {len(test_race):,}")
    print(f"Total annotations: {len(race_map):,}")

    # ------------------------------------------------------------
    # Read identity CSV
    # ------------------------------------------------------------

    rows = []

    with open(
        args.identity,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as infile:

        reader = csv.DictReader(infile)

        # Remove the mysterious None column if present
        fieldnames = [
            field
            for field in reader.fieldnames
            if field is not None
        ]

        # Add our new columns
        fieldnames += [
            "Ethnicity",
            "Ethnicity_Name"
        ]

        missing = 0

        for row in reader:

            # Remove any unexpected fields, including None
            row = {
                key: value
                for key, value in row.items()
                if key in reader.fieldnames and key is not None
            }

            class_id = row["Class_ID"].strip()

            ethnicity = race_map.get(class_id)

            if ethnicity is None:

                row["Ethnicity"] = ""
                row["Ethnicity_Name"] = ""

                missing += 1

            else:

                row["Ethnicity"] = ethnicity
                row["Ethnicity_Name"] = ETHNICITY_NAMES.get(
                    ethnicity,
                    "Unknown"
                )

            rows.append(row)

    # ------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------

    with open(
        args.output,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote: {args.output}")
    print(f"Identity rows: {len(rows):,}")
    print(f"Missing ethnicity labels: {missing:,}")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    counts = {}

    for row in rows:

        ethnicity = row["Ethnicity"]

        if ethnicity:

            counts[ethnicity] = counts.get(
                ethnicity,
                0
            ) + 1

    print()
    print("Identity counts:")

    for number in sorted(counts):

        name = ETHNICITY_NAMES.get(
            number,
            "Unknown"
        )

        print(
            f"  {number}: {name}: {counts[number]:,}"
        )


if __name__ == "__main__":
    main()
