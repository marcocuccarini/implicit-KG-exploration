import json
from collections import defaultdict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


RESULTS_PATH = "result/implicit_results_gemma3_4b.json"

KG_SOURCES = ["wikidata", "conceptnet", "local"]
ALL_SOURCES = KG_SOURCES + ["all"]


# -------------------------
# Triple → readable text
# -------------------------
def triple_to_text(triple):

    s = triple.get("subject", "")
    p = triple.get("predicate", "")
    o = triple.get("object", "")

    if p.startswith("http"):
        p = p.split("/")[-1]

    return f"{s} {p} {o}."


# -------------------------
# Filter triples by KG
# -------------------------
def filter_triples(triples, source):

    if source == "all":
        return triples

    return [
        t for t in triples
        if t.get("source") == source
    ]


# -------------------------
# Build datasets per step AND per KG
# -------------------------
def build_datasets(results):

    datasets = defaultdict(lambda: defaultdict(lambda: {
        "question": [],
        "contexts": [],
        "answer": []
    }))

    for row in results:

        text = row.get("text", "")

        for step in row.get("steps", []):

            step_id = step.get("step")

            explanation = step.get("explanation", "")
            triples = step.get("relevant_triples", [])

            if not explanation:
                continue

            for kg in ALL_SOURCES:

                filtered = filter_triples(triples, kg)

                contexts = [
                    triple_to_text(t)
                    for t in filtered
                ]

                # step 0 special case
                if step_id == 0:
                    contexts = ["No KG used"]

                if not contexts:
                    continue

                datasets[step_id][kg]["question"].append(text)
                datasets[step_id][kg]["contexts"].append(contexts)
                datasets[step_id][kg]["answer"].append(explanation)

    # convert to HuggingFace datasets
    hf_datasets = {}

    for step_id, kg_data in datasets.items():

        hf_datasets[step_id] = {}

        for kg, data in kg_data.items():

            if len(data["question"]) == 0:
                continue

            hf_datasets[step_id][kg] = Dataset.from_dict(data)

            print(f"Step {step_id} | KG {kg} | samples: {len(data['question'])}")

    return hf_datasets


# -------------------------
# Evaluate everything
# -------------------------
def evaluate_all(hf_datasets):

    final_scores = {}

    for step_id in sorted(hf_datasets.keys()):

        final_scores[step_id] = {}

        for kg in hf_datasets[step_id]:

            dataset = hf_datasets[step_id][kg]

            print(f"\nEvaluating STEP {step_id} | KG {kg}")

            scores = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision
                ]
            )

            final_scores[step_id][kg] = scores

            print(scores)

    return final_scores


# -------------------------
# Pretty print results
# -------------------------
def print_summary(scores):

    print("\n\nFINAL SUMMARY TABLE\n")

    for step in sorted(scores.keys()):

        print(f"\nSTEP {step}")

        for kg, metrics in scores[step].items():

            f = metrics.get("faithfulness", 0)
            p = metrics.get("context_precision", 0)
            r = metrics.get("answer_relevancy", 0)

            print(
                f"{kg:10} | "
                f"Faithfulness: {f:.3f} | "
                f"Precision: {p:.3f} | "
                f"Relevancy: {r:.3f}"
            )


# -------------------------
# Main
# -------------------------
def main():

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    hf_datasets = build_datasets(results)

    scores = evaluate_all(hf_datasets)

    print_summary(scores)


if __name__ == "__main__":
    main()
