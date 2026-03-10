import json
import csv
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ----------------------------
# Paths
# ----------------------------
INPUT_DIR = Path("evaluation/automatic_eval")
OUTPUT_DIR = Path("evaluation/automatic_evaluuted_SBERT")
SUMMARY_DIR = Path("evaluation/automatic_evaluuted_SBERT_summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load SBERT model
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Helper functions
# ----------------------------
def parse_json_list(text):
    try:
        return ast.literal_eval(text)
    except:
        return []

def compute_sbert_score(candidate, references):
    if not references:
        return 0.0
    candidate_emb = model.encode(candidate, convert_to_tensor=True)
    ref_embs = model.encode(references, convert_to_tensor=True)
    cosine_scores = util.cos_sim(candidate_emb, ref_embs)
    return float(cosine_scores.mean().item())

def compute_bleu_score(candidate, references):
    if not references:
        return 0.0
    smoothie = SmoothingFunction().method4
    scores = [sentence_bleu([ref.split()], candidate.split(), smoothing_function=smoothie) for ref in references]
    return sum(scores)/len(scores)

# ----------------------------
# Process all CSV files
# ----------------------------
csv_files = list(INPUT_DIR.glob("*.csv"))
master_summary = []

for csv_file in csv_files:
    print(f"\nProcessing {csv_file.name}")
    rows_out = []

    total_metrics = {
        "sbert_baseline": 0.0,
        "sbert_improved_max": 0.0,
        "sbert_improved_min": 0.0,
        "bleu_baseline": 0.0,
        "bleu_improved_max": 0.0,
        "bleu_improved_min": 0.0,
        "combined_avg": 0.0
    }

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        num_rows = 0
        for row in reader:
            num_rows += 1
            implicit_texts = parse_json_list(row.get("implicit_text", "[]"))

            if not implicit_texts:
                avg_sbert_baseline = avg_sbert_max = avg_sbert_min = 0.0
                avg_bleu_baseline = avg_bleu_max = avg_bleu_min = 0.0
            else:
                avg_sbert_baseline = compute_sbert_score(row["baseline_explanation"], implicit_texts)
                avg_sbert_max = compute_sbert_score(row["improved_text_max"], implicit_texts)
                avg_sbert_min = compute_sbert_score(row["improved_text_min"], implicit_texts)

                avg_bleu_baseline = compute_bleu_score(row["baseline_explanation"], implicit_texts)
                avg_bleu_max = compute_bleu_score(row["improved_text_max"], implicit_texts)
                avg_bleu_min = compute_bleu_score(row["improved_text_min"], implicit_texts)

            combined_avg = (avg_sbert_baseline + avg_sbert_max + avg_sbert_min +
                            avg_bleu_baseline + avg_bleu_max + avg_bleu_min) / 6

            total_metrics["sbert_baseline"] += avg_sbert_baseline
            total_metrics["sbert_improved_max"] += avg_sbert_max
            total_metrics["sbert_improved_min"] += avg_sbert_min
            total_metrics["bleu_baseline"] += avg_bleu_baseline
            total_metrics["bleu_improved_max"] += avg_bleu_max
            total_metrics["bleu_improved_min"] += avg_bleu_min
            total_metrics["combined_avg"] += combined_avg

            row.update({
                "sbert_baseline": avg_sbert_baseline,
                "sbert_improved_max": avg_sbert_max,
                "sbert_improved_min": avg_sbert_min,
                "bleu_baseline": avg_bleu_baseline,
                "bleu_improved_max": avg_bleu_max,
                "bleu_improved_min": avg_bleu_min,
                "combined_avg": combined_avg
            })

            rows_out.append(row)

    # Save detailed CSV
    output_file = OUTPUT_DIR / f"{csv_file.stem}_eval.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved evaluated CSV → {output_file}")

    # Compute averages for this file
    averages = {key: total_metrics[key]/num_rows for key in total_metrics}
    averages["dataset"] = csv_file.stem
    master_summary.append(averages)

# ----------------------------
# Save master summary CSV
# ----------------------------
master_summary_file = SUMMARY_DIR / "master_summary.csv"
fieldnames = ["dataset"] + [k for k in total_metrics.keys()]
with open(master_summary_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(master_summary)

print(f"\nSaved master summary CSV → {master_summary_file}")
print("\nAll files processed successfully.")