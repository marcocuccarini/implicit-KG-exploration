import csv
import json
import os
from collections import defaultdict

# Importing your specific configurations and clients
from config import *
from llm.ollama_client import OllamaChat
from kg.local_graph import LocalGraph
from kg.wikidata import WikidataClient
from kg.conceptnet import ConceptNetClient
from kg.explorer import KGExplorer
from pipeline.iterative import iterative_explanation
from utils.normalization import normalize_target_list
from utils.semantic_filter import SemanticTripleFilter

def main():
    # 1. Initialize components
    llm = OllamaChat(LLM_MODEL)
    semantic_filter = SemanticTripleFilter()
    explorer = KGExplorer(
        WikidataClient(CACHE_FILE), 
        ConceptNetClient(), 
        LocalGraph(LOCAL_KG_PATH, STER_URI)
    )

    results = []
    processed_ids = set()

    # 2. Setup Results Directory
    if os.path.dirname(RESULTS_PATH):
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    # 3. Load Progress (Checkpointing)
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                processed_ids = {str(r["id"]) for r in results}
                print(f"Resuming from checkpoint. Already processed {len(processed_ids)} items.")
            except json.JSONDecodeError:
                results = []

    # 4. Load Dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Define the KG sources to be used by the explorer
    active_sources = ["wikidata", "conceptnet", "local"]

    # 5. Main Processing Loop
    cont=0
    for i, row in enumerate(rows, start=1):


        row_id = str(row.get("unique_id", i))

        print("Num riga", cont)

        cont+=1
        
        if row_id in processed_ids: 
            continue

        text = row.get("text", "").strip()
        targets = normalize_target_list(row.get("target", ""))
        
        if not text or not targets:
            print(f"Skipping ID {row_id}: Missing text or targets.")
            continue

        print(f"\n>>> Processing ID {row_id} | Targets: {targets}")

        try:
            # --- FIXED ARGUMENT ORDER ---
            # Order must match: (text_id, text, targets, llm, explorer, sources)
            out = iterative_explanation(
                text_id=row_id,
                text=text,
                targets=targets,
                llm=llm,
                explorer=explorer,
                sources=active_sources,
                semantic_filter=semantic_filter   # new argument
            )

            # 6. Formatting the Trace (Grouping triples by source for the report)
            for step_info in out["steps"]:
                kg_by_source = defaultdict(list)
                
                # Check if relevant_triples exists in this step
                relevant_triples = step_info.get("relevant_triples", [])
                
                for triple in relevant_triples:
                    src = triple.get("source", "unknown")
                    kg_by_source[src].append({
                        "s": triple.get("subject"), 
                        "p": triple.get("predicate"), 
                        "o": triple.get("object")
                    })
                
                # Update the step dictionary with the summarized KG evidence
                step_info["kg_by_source_summary"] = dict(kg_by_source)

                # Console Output for tracking
                print(f"   Step {step_info['step']} | Conf: {step_info['confidence']:.2f} | Evidence: {len(relevant_triples)} triples")

            # 7. Append to results list
            results.append(out)

            # 8. Incremental Save (Saves after every successful row)
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"CRITICAL ERROR on ID {row_id}: {str(e)}")
            import traceback
            traceback.print_exc() # Helps debug exactly where it crashed
            continue 

    print(f"\nDone. Processed total of {len(results)} rows. Saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()