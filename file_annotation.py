import json
import csv
import random
from langdetect import detect, DetectorFactory

# Ensure deterministic language detection
DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# -------- Load JSON data --------
json_file = 'result/implicit_results_gpt-oss_20b.json'
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found.")
    exit()

# -------- Load previous CSV for implicit_text & stereotype --------
prev_csv = 'evaluation/automatic_eval/got-oss_merged.csv'
id_to_implicit = {}
with open(prev_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_to_implicit[row['id']] = {
            'implicit_text': row.get('implicit_text', ''),
            'stereotype': row.get('stereotype', '')
        }

# -------- Collect all valid candidates --------
valid_en = []
valid_it = []

for item in data:
    item_id = item["id"]
    text = item["text"]
    steps_list = item.get("steps", [])

    steps = {s["step"]: s for s in steps_list}
    if 0 not in steps:
        continue

    baseline_conf = steps[0]["confidence"]
    baseline_text = steps[0]["explanation"]

    candidate_steps = [(s["step"], s["confidence"]) for s in steps_list if 1 <= s["step"] <= 5]
    if not candidate_steps:
        continue

    max_conf = max(conf for _, conf in candidate_steps)
    if max_conf <= baseline_conf:
        continue

    best_step = min(step for step, conf in candidate_steps if conf == max_conf)
    improved_text = steps[best_step]["explanation"]

    lang = detect_language(text)
    extra = id_to_implicit.get(item_id, {})
    implicit_text = extra.get('implicit_text', '')
    stereotype = extra.get('stereotype', '')

    # Skip items without implicit_text
    if not implicit_text:
        continue

    record = {
        "id": item_id,
        "text": text,
        "baseline": baseline_text,
        "improved": improved_text,
        "best_step": best_step,
        "implicit_text": implicit_text,
        "stereotype": stereotype
    }

    if lang == "it":
        valid_it.append(record)
    else:
        valid_en.append(record)

# -------- Limit to maximum 50 samples per language --------
sample_en = random.sample(valid_en, min(50, len(valid_en)))
sample_it = random.sample(valid_it, min(50, len(valid_it)))

# -------- Function to create annotation/tracking CSVs --------
def create_annotation_files(records, lang_code):
    random.shuffle(records)
    annotation_rows = []
    tracking_rows = []

    for idx, item in enumerate(records, 1):
        row_label = f"Row_{idx:03d}"
        pair = [('baseline', item['baseline']), ('improved', item['improved'])]
        random.shuffle(pair)

        explanation_A = pair[0][1]
        explanation_B = pair[1][1]

        # Annotation CSV
        annotation_rows.append([
            row_label,
            item['id'],
            item['text'],
            explanation_A,
            explanation_B,
            item['implicit_text'],
            item['stereotype'],
            "", "", "", ""  # 4 annotator columns
        ])

        # Tracking CSV
        tracking_rows.append([
            row_label,
            item['id'],
            "A" if pair[0][0] == "baseline" else "B",
            "A" if pair[0][0] == "improved" else "B",
            item['best_step']
        ])

    # Write files
    annotation_file = f'evaluation/human_eval/Human_Annotation_Task_{lang_code}.csv'
    tracking_file = f'evaluation/human_eval/Tracking_Key_{lang_code}.csv'

    with open(annotation_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Row_ID',
            'Original_ID',
            'Content',
            'Explanation_A',
            'Explanation_B',
            'Implicit_Statement',
            'Stereotype',
            'Annotator_1',
            'Annotator_2',
            'Annotator_3',
            'Annotator_4'
        ])
        writer.writerows(annotation_rows)

    with open(tracking_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Row_ID',
            'Original_ID',
            'Baseline_Location (A/B)',
            'Improved_Location (A/B)',
            'Improved_Step_Number'
        ])
        writer.writerows(tracking_rows)

    print(f"{lang_code.upper()} annotation files created: {len(records)} rows")

# -------- Generate files --------
create_annotation_files(sample_en, 'en')
create_annotation_files(sample_it, 'it')