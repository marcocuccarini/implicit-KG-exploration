import csv
import ast
from pathlib import Path
from classes.ServerOllama import OllamaServer, OllamaChat, LLMResponse

# ===========================
# Helper Function
# ===========================
def ask_llm_judge(chat_session: OllamaChat, input_text, explanation_a, explanation_b):
    prompt = f"""
You are a strict evaluator of explanations for hateful or biased messages.
Compare two candidate explanations (A and B) with respect to the input sentence.
Evaluate each along four dimensions using a binary score (1 if the statement is true, 0 otherwise):

1. Linguistic Fluency
2. Coherence
3. Relevance
4. Truthfulness

IMPORTANT: Output ONLY valid JSON in the format below. Do not add any extra text:

{{
  "Explanation_A": {{"fluency": int, "coherence": int, "relevance": int, "truthfulness": int}},
  "Explanation_B": {{"fluency": int, "coherence": int, "relevance": int, "truthfulness": int}}
}}

Input Sentence: "{input_text}"
Explanation A: "{explanation_a}"
Explanation B: "{explanation_b}"
"""

    try:
        llm_response: LLMResponse = chat_session.send_prompt(prompt)
        text = llm_response.raw_text.strip()

        llm_scores = ast.literal_eval(text)

        if "Explanation_A" not in llm_scores or "Explanation_B" not in llm_scores:
            raise ValueError("Missing Explanation_A or Explanation_B")

        return llm_scores, llm_response.raw_text

    except Exception as e:
        print(f"\n⚠️ Failed to parse LLM output: {e}")

        fallback = {
            "Explanation_A": {"fluency": 0, "coherence": 0, "relevance": 0, "truthfulness": 0},
            "Explanation_B": {"fluency": 0, "coherence": 0, "relevance": 0, "truthfulness": 0},
        }

        return fallback, llm_response.raw_text if 'llm_response' in locals() else ""

# ===========================
# Main Evaluation Script
# ===========================
INPUT_DIR = Path("evaluation/automatic_eval")
OUTPUT_DIR = Path("evaluation/LLM_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

server = OllamaServer()
chat_session = OllamaChat(server, model="gemma3:27b")

csv_files = list(INPUT_DIR.glob("*.csv"))

for csv_file in csv_files:

    print(f"\nProcessing file: {csv_file.name}")

    rows_out = []

    # Read all rows first so we know dataset size
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)

    print(f"Total rows to evaluate: {total_rows}\n")

    for i, row in enumerate(rows, start=1):

        input_text = row.get("text", "")
        explanation_a = row.get("baseline_explanation", "")
        explanation_b = row.get("improved_text_max", "")

        llm_scores, raw_output = ask_llm_judge(
            chat_session,
            input_text,
            explanation_a,
            explanation_b
        )

        for key in ["fluency", "coherence", "relevance", "truthfulness"]:
            row[f"A_{key}"] = llm_scores["Explanation_A"][key]
            row[f"B_{key}"] = llm_scores["Explanation_B"][key]

        row["A_total"] = sum(llm_scores["Explanation_A"].values())
        row["B_total"] = sum(llm_scores["Explanation_B"].values())

        row["LLM_raw_output"] = raw_output

        rows_out.append(row)

        # ===== Progress display =====
        percent = (i / total_rows) * 100
        print(f"\rProgress: {i}/{total_rows} ({percent:.2f}%)", end="")

    print("\nSaving results...")

    output_file = OUTPUT_DIR / f"{csv_file.stem}_LLM_eval.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved evaluated CSV → {output_file}")

print("\nAll files processed successfully.")