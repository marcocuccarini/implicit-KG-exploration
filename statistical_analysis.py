import os
import json
import statistics
import csv
from collections import Counter, defaultdict

# =========================
# CONFIGURATION VARIABLES
# =========================
INPUT_DIR = "result"
OUTPUT_DIR = "reports"
# =========================


def analyze_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)

    unique_ids = set()
    id_counter = Counter()

    step_counts = []
    reached_threshold_count = 0
    max_step_index = 0

    confidences = []
    step_confidences = defaultdict(list)

    triple_count_total = 0
    triple_count_per_step = []

    triple_type_counter = Counter()
    triple_objects_counter = Counter()

    # NEW: KG usage per step
    kg_type_per_step = defaultdict(Counter)

    target_counter = Counter()
    multi_target_count = 0
    entries_zero_triples = 0

    warnings = []

    for entry in data:

        entry_id = entry.get("id")

        if entry_id is not None:
            unique_ids.add(entry_id)
            id_counter[entry_id] += 1

        conf = entry.get("final_confidence", 0)
        confidences.append(conf)

        targets = entry.get("target", [])

        if len(targets) > 1:
            multi_target_count += 1

        for t in targets:
            target_counter[t] += 1

        steps = entry.get("steps", [])
        step_counts.append(len(steps))

        if len(steps) == 0:
            warnings.append(f"Entry {entry_id} has no steps.")

        entry_triple_count = 0

        for step in steps:

            step_index = step.get("step", 0)
            max_step_index = max(max_step_index, step_index)

            step_conf = step.get("confidence", 0)
            step_confidences[step_index].append(step_conf)

            if step.get("reached_threshold", False):
                reached_threshold_count += 1

            triples = step.get("relevant_triples", [])

            entry_triple_count += len(triples)
            triple_count_total += len(triples)
            triple_count_per_step.append(len(triples))

            for triple in triples:

                source = triple.get("source", "unknown")

                # global counter
                triple_type_counter[source] += 1

                # NEW: per step KG counter
                kg_type_per_step[step_index][source] += 1

                triple_objects_counter[
                    triple.get("object", "unknown")
                ] += 1

        if entry_triple_count == 0:
            entries_zero_triples += 1
            warnings.append(f"Entry {entry_id} has 0 triples.")

        if conf == 0:
            warnings.append(f"Entry {entry_id} has 0 confidence.")

    # =====================
    # Derived statistics
    # =====================

    avg_steps = statistics.mean(step_counts) if step_counts else 0

    avg_conf = statistics.mean(confidences) if confidences else 0
    min_conf = min(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0
    std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0

    threshold_percentage = (
        reached_threshold_count / total_entries * 100
        if total_entries > 0 else 0
    )

    avg_triples_per_entry = (
        triple_count_total / total_entries
        if total_entries > 0 else 0
    )

    avg_triples_per_step = (
        sum(triple_count_per_step) / len(triple_count_per_step)
        if triple_count_per_step else 0
    )

    # Average confidence per step
    avg_conf_per_step = {
        k: statistics.mean(v) if v else 0
        for k, v in step_confidences.items()
    }

    # =====================
    # Confidence bins
    # =====================

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    conf_bins = Counter()

    for c in confidences:

        for i in range(len(bins) - 1):

            if bins[i] <= c <= bins[i + 1]:
                conf_bins[f"{bins[i]}-{bins[i+1]}"] += 1
                break

    duplicate_ids = {
        k: v for k, v in id_counter.items() if v > 1
    }

    kg_types_per_step_dict = {
        step: dict(counter)
        for step, counter in kg_type_per_step.items()
    }

    # =====================
    # Report structure
    # =====================

    report = {

        "file_name": os.path.basename(filepath),

        "total_entries": total_entries,
        "unique_ids": len(unique_ids),
        "duplicate_ids": duplicate_ids,

        "step_statistics": {
            "average_steps_per_entry": avg_steps,
            "max_step_index_observed": max_step_index,
            "average_confidence_per_step": avg_conf_per_step
        },

        "threshold_statistics": {
            "reached_threshold_count": reached_threshold_count,
            "not_reached_threshold_count": total_entries - reached_threshold_count,
            "percentage_reached": threshold_percentage
        },

        "confidence_statistics": {
            "average_confidence": avg_conf,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
            "confidence_std_dev": std_conf,
            "confidence_bins": dict(conf_bins)
        },

        "triple_statistics": {
            "total_triples": triple_count_total,
            "average_triples_per_entry": avg_triples_per_entry,
            "average_triples_per_step": avg_triples_per_step,
            "entries_with_zero_triples": entries_zero_triples,
            "triple_types_by_source": dict(triple_type_counter),
            "kg_types_per_step": kg_types_per_step_dict,
            "top_10_triple_objects": triple_objects_counter.most_common(10)
        },

        "target_distribution": dict(target_counter),
        "multi_target_entries_count": multi_target_count,
        "warnings": warnings
    }

    return report, kg_types_per_step_dict, avg_conf_per_step


def save_reports(report, kg_per_step, avg_conf_per_step, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    base_name = report["file_name"].replace(".json", "")

    # JSON report
    json_path = os.path.join(output_dir, f"{base_name}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # TXT report
    txt_path = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:

        for key, value in report.items():
            f.write(f"{key}:\n")
            f.write(f"{json.dumps(value, indent=4)}\n\n")

    # CSV summary
    csv_path = os.path.join(output_dir, f"{base_name}_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["metric", "value"])

        writer.writerow(["total_entries", report["total_entries"]])
        writer.writerow(["unique_ids", report["unique_ids"]])

        writer.writerow([
            "average_steps_per_entry",
            report["step_statistics"]["average_steps_per_entry"]
        ])

        writer.writerow([
            "average_confidence",
            report["confidence_statistics"]["average_confidence"]
        ])

        writer.writerow([
            "total_triples",
            report["triple_statistics"]["total_triples"]
        ])

    # ======================
    # KG types per step CSV
    # ======================

    kg_csv = os.path.join(
        output_dir,
        f"{base_name}_kg_types_per_step.csv"
    )

    all_sources = set()

    for counter in kg_per_step.values():
        all_sources.update(counter.keys())

    sources = sorted(all_sources)

    with open(kg_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["step"] + sources)

        for step in sorted(kg_per_step.keys()):

            row = [step]

            for s in sources:
                row.append(kg_per_step[step].get(s, 0))

            writer.writerow(row)

    # ======================
    # Confidence per step CSV
    # ======================

    conf_csv = os.path.join(
        output_dir,
        f"{base_name}_confidence_per_step.csv"
    )

    with open(conf_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["step", "average_confidence"])

        for step in sorted(avg_conf_per_step.keys()):
            writer.writerow([step, avg_conf_per_step[step]])


def analyze_directory(input_dir, output_dir):

    for filename in os.listdir(input_dir):

        if filename.endswith(".json"):

            filepath = os.path.join(input_dir, filename)

            print(f"\nAnalyzing {filename}...")

            report, kg_per_step, avg_conf_per_step = analyze_file(filepath)

            save_reports(
                report,
                kg_per_step,
                avg_conf_per_step,
                output_dir
            )

    print("\nAll reports generated.")


if __name__ == "__main__":
    analyze_directory(INPUT_DIR, OUTPUT_DIR)