import csv
import json
from pathlib import Path
from openai import OpenAI

# ===========================
# Configuration
# ===========================

client = OpenAI(api_key="")  # usa OPENAI_API_KEY dalle env vars

INPUT_DIR = Path("evaluation/automatic_eval")
OUTPUT_DIR = Path(".")  # salva nella stessa posizione dello script

# ===========================
# Helper Function
# ===========================

def ask_llm_judge(input_text, explanation_a, explanation_b):

    prompt = f"""
You are an expert evaluator of explanations for hateful or biased messages.

Compare two candidate explanations (A and B) based on how well they are grounded in the input text.

SCORING SCALE:
0 - The response is not consistent with the information contained in the input text.
1 - Although the implied assumption generated is correct or partially correct, the response is generic or does not contain any explicit or implicit reference to what it has been said in the input text.
2 - The implied assumption generated is partially correct and grounded in the input text.
3 - The implied assumption is correct and grounded in the input text.

Input Sentence: "{input_text}"
Explanation A: "{explanation_a}"
Explanation B: "{explanation_b}"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        llm_scores = json.loads(raw_content)

        if "Explanation_A" not in llm_scores or "Explanation_B" not in llm_scores:
            raise ValueError("Missing keys in LLM JSON response")

        return llm_scores, raw_content

    except Exception as e:
        print(f"\n⚠️ OpenAI API Error: {e}")
        fallback = {
            "Explanation_A": {"score": 0},
            "Explanation_B": {"score": 0},
        }
        return fallback, str(e)


# ===========================
# Main Evaluation Script
# ===========================

csv_files = list(INPUT_DIR.glob("*.csv"))

for csv_file in csv_files:

    print(f"\nProcessing file: {csv_file.name}")

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)

    print(f"Total rows to evaluate: {total_rows}\n")

    output_file = OUTPUT_DIR / f"{csv_file.stem}_LLM_eval.csv"

    # Apri il file in modalità scrittura
    with open(output_file, "w", newline="", encoding="utf-8") as out_f:

        fieldnames = list(rows[0].keys()) + [
            "A_groundedness_score",
            "B_groundedness_score",
            "LLM_raw_output"
        ]

        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):

            input_text = row.get("text", "")
            explanation_a = row.get("baseline_explanation", "")
            explanation_b = row.get("improved_text_max", "")

            llm_scores, raw_output = ask_llm_judge(
                input_text,
                explanation_a,
                explanation_b
            )

            row["A_groundedness_score"] = llm_scores["Explanation_A"].get("score", 0)
            row["B_groundedness_score"] = llm_scores["Explanation_B"].get("score", 0)
            row["LLM_raw_output"] = raw_output

            writer.writerow(row)

            # salva subito sul disco
            out_f.flush()

            percent = (i / total_rows) * 100

            print(f"\rProgress: {i}/{total_rows} ({percent:.2f}%)", end="")

    print(f"\nSaved evaluated CSV → {output_file}")

print("\nAll files processed successfully.")