import json
import csv
import random

# -----------------------------
# Load main JSON
# -----------------------------
try:
    with open('result/implicit_results_gpt-oss_20b.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: main JSON file not found.")
    exit()

# -----------------------------
# Load second baseline JSON
# -----------------------------
try:
    with open('result/baseline_gpt-oss_20b.json', 'r', encoding='utf-8') as f:
        baseline2_data = json.load(f)
except FileNotFoundError:
    print("Error: second baseline JSON file not found.")
    exit()

baseline2_dict = {item["id"]: item for item in baseline2_data}

annotation_rows = []
tracking_rows = []

row_counter = 1

for item in data:

    item_id = item["id"]
    text = item["text"]
    steps_list = item.get("steps", [])

    steps = {s["step"]: s for s in steps_list}

    if 0 not in steps:
        continue

    # Baseline 1
    baseline1_text = steps[0]["explanation"]
    baseline1_conf = steps[0]["confidence"]

    # Baseline 2
    baseline2 = baseline2_dict.get(item_id)
    if not baseline2:
        continue

    baseline2_text = baseline2["explanation"]
    baseline2_conf = baseline2["confidence"]

    # Candidate improved steps
    candidate_steps = [
        (s["step"], s["confidence"])
        for s in steps_list
        if 1 <= s["step"] <= 5
    ]

    if not candidate_steps:
        continue

    max_conf = max(conf for _, conf in candidate_steps)

    # Improved must beat both baselines
    if max_conf <= max(baseline1_conf, baseline2_conf):
        continue

    best_step = min(
        step for step, conf in candidate_steps if conf == max_conf
    )

    improved_text = steps[best_step]["explanation"]

    if row_counter > 50:
        break

    row_label = f"Row_{row_counter:02d}"

    # ---------------------------------
    # Randomize A/B/C (blind)
    # ---------------------------------
    explanations = [
        ("baseline1", baseline1_text),
        ("baseline2", baseline2_text),
        ("improved", improved_text)
    ]

    random.shuffle(explanations)

    explanation_A = explanations[0][1]
    explanation_B = explanations[1][1]
    explanation_C = explanations[2][1]

    # Annotation file
    annotation_rows.append([
        row_label,
        item_id,
        text,
        explanation_A,
        explanation_B,
        explanation_C,
        "", "", "", ""
    ])

    # Tracking file
    mapping = {label: pos for pos, (label, _) in zip(["A","B","C"], explanations)}

    tracking_rows.append([
        row_label,
        item_id,
        mapping["baseline1"],
        mapping["baseline2"],
        mapping["improved"],
        best_step
    ])

    row_counter += 1


if row_counter <= 50:
    print(f"Warning: Only found {row_counter-1} valid cases.")

# -----------------------------
# Write annotation file
# -----------------------------
with open('evaluation/human_eval/Human_Annotation_Task.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Row_ID',
        'Original_ID',
        'Content',
        'Explanation_A',
        'Explanation_B',
        'Explanation_C',
        'Annotator_1',
        'Annotator_2',
        'Annotator_3',
        'Annotator_4'
    ])
    writer.writerows(annotation_rows)

# -----------------------------
# Write tracking file
# -----------------------------
with open('evaluation/human_eval/Tracking_Key.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Row_ID',
        'Original_ID',
        'Baseline1_Position',
        'Baseline2_Position',
        'Improved_Position',
        'Improved_Step_Number'
    ])
    writer.writerows(tracking_rows)

print("Success! Blind annotation file created.")