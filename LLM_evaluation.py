import csv
import json
from pathlib import Path
from openai import OpenAI

# ===========================
# Configuration
# ===========================
# Initialize the OpenAI client (it will look for OPENAI_API_KEY in your env vars)
client = OpenAI(api_key="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") 

# ===========================
# Helper Function
# ===========================
def ask_llm_judge(input_text, explanation_a, explanation_b):
    # Prompt using the 0-3 groundedness scale
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
            model="gpt-4o", # Or "gpt-4o-mini" for faster/cheaper processing
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Forces JSON output
        )

        raw_content = response.choices[0].message.content
        llm_scores = json.loads(raw_content)

        # Validate structure
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
INPUT_DIR = Path("evaluation/automatic_eval")
OUTPUT_DIR = Path("evaluation/LLM_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(INPUT_DIR.glob("*.csv"))

for csv_file in csv_files:
    print(f"\nProcessing file: {csv_file.name}")
    rows_out = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)
    print(f"Total rows to evaluate: {total_rows}\n")

    for i, row in enumerate(rows, start=1):
        input_text = row.get("text", "")
        explanation_a = row.get("baseline_explanation", "")
        explanation_b = row.get("improved_text_max", "")

        llm_scores, raw_output = ask_llm_judge(
            input_text,
            explanation_a,
            explanation_b
        )

        # Add the results to the row
        row["A_groundedness_score"] = llm_scores["Explanation_A"].get("score", 0)
        row["B_groundedness_score"] = llm_scores["Explanation_B"].get("score", 0)
        row["LLM_raw_output"] = raw_output

        rows_out.append(row)

        # Progress display
        percent = (i / total_rows) * 100
        print(f"\rProgress: {i}/{total_rows} ({percent:.2f}%)", end="")

    print("\nSaving results...")
    output_file = OUTPUT_DIR / f"{csv_file.stem}_LLM_eval.csv"

    if rows_out:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows_out[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"Saved evaluated CSV → {output_file}")

print("\nAll files processed successfully.")