import json
import csv
import random

# Load JSON data
try:
    with open('result/implicit_results_gpt-oss_20b.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found.")
    exit()

annotation_rows = []
tracking_rows = []

row_counter = 1

for item in data:

    item_id = item["id"]
    text = item["text"]
    steps_list = item.get("steps", [])

    # Convert list to dict
    steps = {s["step"]: s for s in steps_list}

    if 0 not in steps:
        continue

    baseline_conf = steps[0]["confidence"]
    baseline_text = steps[0]["explanation"]

    # Candidate steps 1–5
    candidate_steps = [
        (s["step"], s["confidence"])
        for s in steps_list
        if 1 <= s["step"] <= 5
    ]

    if not candidate_steps:
        continue

    max_conf = max(conf for _, conf in candidate_steps)

    # Only if better than baseline
    if max_conf <= baseline_conf:
        continue

    # Tie-breaking → lowest step
    best_step = min(
        step for step, conf in candidate_steps if conf == max_conf
    )

    improved_text = steps[best_step]["explanation"]

    if row_counter > 50:
        break

    row_label = f"Row_{row_counter:02d}"

    # 🔀 Randomize order (BLINDING)
    pair = [("baseline", baseline_text), ("improved", improved_text)]
    random.shuffle(pair)

    explanation_A = pair[0][1]
    explanation_B = pair[1][1]

    # Annotation file (annotator cannot know)
    annotation_rows.append([
        row_label,
        item_id,
        text,
        explanation_A,
        explanation_B,
        "", "", "", ""  # 4 annotators
    ])

    # Tracking file (you know the truth)
    tracking_rows.append([
        row_label,
        item_id,
        "A" if pair[0][0] == "baseline" else "B",
        "A" if pair[0][0] == "improved" else "B",
        best_step
    ])

    row_counter += 1

if row_counter <= 50:
    print(f"Warning: Only found {row_counter-1} valid cases.")

# Write Annotation File (BLIND)
with open('Human_Annotation_Task.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Row_ID',
        'Original_ID',
        'Content',
        'Explanation_A',
        'Explanation_B',
        'Annotator_1',
        'Annotator_2',
        'Annotator_3',
        'Annotator_4'
    ])
    writer.writerows(annotation_rows)

# Write Tracking File (PRIVATE – do not share)
with open('Tracking_Key.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Row_ID',
        'Original_ID',
        'Baseline_Location (A/B)',
        'Improved_Location (A/B)',
        'Improved_Step_Number'
    ])
    writer.writerows(tracking_rows)

print("Success! Blind annotation file created.")