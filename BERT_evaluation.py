import csv
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# ----------------------------
# Initialize ROUGE scorer
# ----------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# ----------------------------
# Helper functions
# ----------------------------
def parse_json_list(text):
    try:
        return ast.literal_eval(text)
    except:
        return []

def compute_sbert_score(candidate, references):
    if not candidate or not references:
        return 0.0

    candidate_emb = model.encode(candidate, convert_to_tensor=True)
    ref_embs = model.encode(references, convert_to_tensor=True)

    cosine_scores = util.cos_sim(candidate_emb, ref_embs)
    return float(cosine_scores.mean().item())

def compute_bleu_score(candidate, references):
    if not candidate or not references:
        return 0.0

    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([ref.split()], candidate.split(), smoothing_function=smoothie)
        for ref in references
    ]
    return sum(scores) / len(scores)

def compute_rouge_score(candidate, references):
    if not candidate or not references:
        return 0.0

    scores = []
    for ref in references:
        result = scorer.score(ref, candidate)
        scores.append(result["rougeL"].fmeasure)

    return sum(scores) / len(scores)

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
        "sbert_target": 0.0,
        "sbert_improved_max": 0.0,
        "sbert_improved_min": 0.0,
        "bleu_baseline": 0.0,
        "bleu_target": 0.0,
        "bleu_improved_max": 0.0,
        "bleu_improved_min": 0.0,
        "rouge_baseline": 0.0,
        "rouge_target": 0.0,
        "rouge_improved_max": 0.0,
        "rouge_improved_min": 0.0,
        "combined_avg": 0.0
    }

    with open(csv_file, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        num_rows = 0

        for row in reader:

            num_rows += 1

            implicit_texts = parse_json_list(row.get("implicit_text", "[]"))

            baseline_explanation = row.get("baseline_explanation", "")
            original_text = row.get("text", "")
            improved_max = row.get("improved_text_max", "")
            improved_min = row.get("improved_text_min", "")

            # ----------------------------
            # SBERT
            # ----------------------------

            avg_sbert_baseline = compute_sbert_score(baseline_explanation, implicit_texts)
            avg_sbert_target = compute_sbert_score(original_text, implicit_texts)
            avg_sbert_max = compute_sbert_score(improved_max, implicit_texts)
            avg_sbert_min = compute_sbert_score(improved_min, implicit_texts)

            # ----------------------------
            # BLEU
            # ----------------------------

            avg_bleu_baseline = compute_bleu_score(baseline_explanation, implicit_texts)
            avg_bleu_target = compute_bleu_score(original_text, implicit_texts)
            avg_bleu_max = compute_bleu_score(improved_max, implicit_texts)
            avg_bleu_min = compute_bleu_score(improved_min, implicit_texts)

            # ----------------------------
            # ROUGE
            # ----------------------------

            avg_rouge_baseline = compute_rouge_score(baseline_explanation, implicit_texts)
            avg_rouge_target = compute_rouge_score(original_text, implicit_texts)
            avg_rouge_max = compute_rouge_score(improved_max, implicit_texts)
            avg_rouge_min = compute_rouge_score(improved_min, implicit_texts)

            # ----------------------------
            # Combined score
            # ----------------------------

            combined_avg = (
                avg_sbert_baseline + avg_sbert_target + avg_sbert_max + avg_sbert_min +
                avg_bleu_baseline + avg_bleu_target + avg_bleu_max + avg_bleu_min +
                avg_rouge_baseline + avg_rouge_target + avg_rouge_max + avg_rouge_min
            ) / 12

            # ----------------------------
            # Update totals
            # ----------------------------

            total_metrics["sbert_baseline"] += avg_sbert_baseline
            total_metrics["sbert_target"] += avg_sbert_target
            total_metrics["sbert_improved_max"] += avg_sbert_max
            total_metrics["sbert_improved_min"] += avg_sbert_min
            total_metrics["bleu_baseline"] += avg_bleu_baseline
            total_metrics["bleu_target"] += avg_bleu_target
            total_metrics["bleu_improved_max"] += avg_bleu_max
            total_metrics["bleu_improved_min"] += avg_bleu_min
            total_metrics["rouge_baseline"] += avg_rouge_baseline
            total_metrics["rouge_target"] += avg_rouge_target
            total_metrics["rouge_improved_max"] += avg_rouge_max
            total_metrics["rouge_improved_min"] += avg_rouge_min
            total_metrics["combined_avg"] += combined_avg

            # ----------------------------
            # Update row
            # ----------------------------

            row.update({
                "sbert_baseline": avg_sbert_baseline,
                "sbert_target": avg_sbert_target,
                "sbert_improved_max": avg_sbert_max,
                "sbert_improved_min": avg_sbert_min,
                "bleu_baseline": avg_bleu_baseline,
                "bleu_target": avg_bleu_target,
                "bleu_improved_max": avg_bleu_max,
                "bleu_improved_min": avg_bleu_min,
                "rouge_baseline": avg_rouge_baseline,
                "rouge_target": avg_rouge_target,
                "rouge_improved_max": avg_rouge_max,
                "rouge_improved_min": avg_rouge_min,
                "combined_avg": combined_avg
            })

            rows_out.append(row)

    # ----------------------------
    # Save evaluated CSV
    # ----------------------------

    if not rows_out:
        continue

    output_file = OUTPUT_DIR / f"{csv_file.stem}_eval.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:

        fieldnames = list(rows_out[0].keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved evaluated CSV → {output_file}")

    # ----------------------------
    # Compute averages
    # ----------------------------

    averages = {key: total_metrics[key] / num_rows for key in total_metrics}

    averages["dataset"] = csv_file.stem

    master_summary.append(averages)

# ----------------------------
# Save master summary
# ----------------------------

master_summary_file = SUMMARY_DIR / "master_summary.csv"

fieldnames = ["dataset"] + [k for k in total_metrics.keys()]

with open(master_summary_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(master_summary)

print(f"\nSaved master summary CSV → {master_summary_file}")
print("\nAll files processed successfully.")