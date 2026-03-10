import csv
import json
import os
import time
from itertools import islice
from config import * 
from llm.ollama_client import OllamaChat
from utils.normalization import normalize_target_list
from utils.json_utils import safe_json_load

# 1. Configuration
BASELINE_MODELS = ["gemma3:4b", "gpt-oss:20b", "qwen3:8b"]
SAMPLE_LIMIT = 2000 

def get_baseline_prompt(text, targets):
    targets_str = ", ".join(targets)
    return f"""
ROLE: Sociolinguistic Auditor
TEXT: \"\"\"{text}\"\"\"
TARGETS: {targets_str}

TASK: Identify if the text contains an implicit stereotype or bias.
Return ONLY valid JSON:
{{ 
  "explanation": "string", 
  "confidence": float 
}}
"""

def run_benchmarks():
    os.makedirs("results", exist_ok=True)

    for model_name in BASELINE_MODELS:
        print(f"\n>>> [STARTING] Model: {model_name} (Limit: {SAMPLE_LIMIT})")
        llm = OllamaChat(model_name)
        model_results = []
        
        # Define the unique file for this model
        safe_name = model_name.replace(":", "_")
        file_path = f"results/baseline_{safe_name}.json"

        with open(DATASET_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Use islice to grab only the first 2000 rows
            dataset_slice = islice(reader, SAMPLE_LIMIT)

            for i, row in enumerate(dataset_slice, start=1):
                text = row.get("text", "").strip()
                targets = normalize_target_list(row.get("target", ""))
                
                if not text: continue

                print(f"  [{model_name}] Processing {i}/{SAMPLE_LIMIT}...", end="\r")
                
                start_time = time.time()
                raw_res = llm.send_prompt(get_baseline_prompt(text, targets))
                latency = round(time.time() - start_time, 2)
                
                data = safe_json_load(raw_res) or {}

                model_results.append({
                    "id": row.get("unique_id", str(i)),
                    "explanation": data.get("explanation", ""),
                    "confidence": data.get("confidence", 0.0),
                    "latency": latency
                })

        # Save the dedicated file for this model
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDONE: Saved {len(model_results)} results to {file_path}")

if __name__ == "__main__":
    run_benchmarks()