import json
import csv
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
INPUT_DIR = Path("result")
OUTPUT_DIR = Path("evaluation/automatic_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CSV = Path("data/data/dataset_split_test.csv")

# ----------------------------
# Load reference CSV by unique_id
# ----------------------------
unique_id_lookup = {}

with open(REFERENCE_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        unique_id_lookup[row["unique_id"]] = {
            "implicit_text": row.get("text_implied", "[]"),
            "stereotype": row.get("stereotype", "[]")
        }

# ----------------------------
# Load all JSON files
# ----------------------------
json_files = list(INPUT_DIR.glob("*.json"))

all_data = {}
all_id_sets = []

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_data[json_file] = data
    ids = {str(item["id"]) for item in data}
    all_id_sets.append(ids)

# ----------------------------
# Find common IDs across files
# ----------------------------
common_ids = set.intersection(*all_id_sets)

print(f"Total common samples across all JSONs: {len(common_ids)}")

# ----------------------------
# Process JSON files
# ----------------------------
for json_file, data in all_data.items():

    print(f"\nProcessing {json_file.name}")

    rows = []

    for item in data:

        item_id = str(item["id"])

        # Skip items not in shared sample set
        if item_id not in common_ids:
            continue

        text = item["text"]

        if item_id in unique_id_lookup:
            implicit_text = unique_id_lookup[item_id]["implicit_text"]
            stereotype = unique_id_lookup[item_id]["stereotype"]
        else:
            implicit_text = "[]"
            stereotype = "[]"

        steps_list = item.get("steps", [])

        steps = {}
        for s in steps_list:
            try:
                step_num = int(s["step"])
                steps[step_num] = s
            except:
                continue

        if 0 not in steps:
            continue

        baseline_conf = steps[0]["confidence"]
        baseline_text = steps[0]["explanation"]

        candidate_steps = [(step, s["confidence"]) for step, s in steps.items() if step != 0]

        # ----------------------------
        # Max tie-break
        # ----------------------------
        if not candidate_steps:
            best_step_max = 0
        else:
            max_conf = max(conf for _, conf in candidate_steps)
            if max_conf < baseline_conf:
                best_step_max = 0
            elif max_conf == baseline_conf:
                best_step_max = max(step for step, conf in candidate_steps if conf == max_conf)
            else:
                best_step_max = max(step for step, conf in candidate_steps if conf == max_conf)

        improved_text_max = steps[best_step_max]["explanation"]
        improved_conf_max = steps[best_step_max]["confidence"]

        # ----------------------------
        # Min tie-break
        # ----------------------------
        if not candidate_steps:
            best_step_min = 0
        else:
            max_conf = max(conf for _, conf in candidate_steps)
            if max_conf < baseline_conf:
                best_step_min = 0
            elif max_conf == baseline_conf:
                best_step_min = min(step for step, conf in candidate_steps if conf == max_conf)
            else:
                best_step_min = min(step for step, conf in candidate_steps if conf == max_conf)

        improved_text_min = steps[best_step_min]["explanation"]
        improved_conf_min = steps[best_step_min]["confidence"]

        is_improved = 1 if best_step_max != 0 else 0

        rows.append([
            item_id,
            text,
            baseline_text,
            baseline_conf,
            best_step_max,
            improved_text_max,
            improved_conf_max,
            best_step_min,
            improved_text_min,
            improved_conf_min,
            is_improved,
            implicit_text,
            stereotype
        ])

    # ----------------------------
    # Save CSV
    # ----------------------------
    base_name = json_file.stem
    output_file = OUTPUT_DIR / f"{base_name}_auto_eval_tiebreak_modes.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "text",
            "baseline_explanation",
            "baseline_confidence",
            "improved_step_max",
            "improved_text_max",
            "improved_conf_max",
            "improved_step_min",
            "improved_text_min",
            "improved_conf_min",
            "is_improved",
            "implicit_text",
            "stereotype"
        ])

        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {output_file}")

print("\nAll files processed successfully.")