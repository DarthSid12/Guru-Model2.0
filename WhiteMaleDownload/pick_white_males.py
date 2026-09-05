import csv
import random

INPUT_FILE = "identity_meta_race.csv"
OUTPUT_FILE = "white_male_id_nums.txt"

# Read CSV explicitly
with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)

    # First row is the header
    header = next(reader)

    # Remove whitespace from column names
    header = [column.strip() for column in header]

    print("CSV columns found:")
    print(header)

    # Find the columns we need
    class_id_index = header.index("Class_ID")
    gender_index = header.index("Gender")
    ethnicity_index = header.index("Ethnicity")

    # Store all Caucasian Latin male identities
    latin_white_males = []

    for row in reader:

        # Skip malformed/empty rows
        if len(row) <= max(class_id_index, gender_index, ethnicity_index):
            continue

        class_id = row[class_id_index].strip()
        gender = row[gender_index].strip().lower()
        ethnicity = row[ethnicity_index].strip()

        if gender == "m" and ethnicity == "3":
            latin_white_males.append(class_id)


# Make sure there are enough candidates
if len(latin_white_males) < 64:
    raise ValueError(
        f"Only {len(latin_white_males)} Caucasian Latin males found; "
        f"cannot select 64."
    )

# Pick 64 unique random indices
random_indices = random.sample(
    range(len(latin_white_males)),
    64
)

# Get the corresponding identity numbers
selected_ids = [
    latin_white_males[i]
    for i in random_indices
]

# Write IDs only, one per line
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for identity_id in selected_ids:
        f.write(identity_id + "\n")


print()
print(f"Found {len(latin_white_males)} Caucasian Latin males.")
print("Selected 64 random identities.")
print(f"Saved to: {OUTPUT_FILE}")
