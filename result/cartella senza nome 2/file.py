import json

file_pairs = [
    ("implicit_results_gemma3_4b_en.json", "implicit_results_gemma3_4b_it.json"),
    ("implicit_results_qwen3_8b_en.json", "implicit_results_qwen3_8b_it.json"),
    ("implicit_results_gpt-oss_20b_en.json", "implicit_results_gpt-oss_20b_it.json"),
]

for en_file, it_file in file_pairs:
    with open(en_file, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    with open(it_file, "r", encoding="utf-8") as f:
        it_data = json.load(f)

    it_dict = {sample["id"]: sample for sample in it_data}

    italian_count = 0
    english_count = 0
    merged = []

    for sample in en_data:
        sid = sample["id"]
        if sid in it_dict:
            merged.append(it_dict[sid])
            italian_count += 1
        else:
            merged.append(sample)
            english_count += 1

    base_name = en_file.replace("_en.json", ".json")

    with open(base_name, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = italian_count + english_count

    print(f"\nFile: {base_name}")
    print(f"Total samples: {total}")
    print(f"Italian samples: {italian_count}")
    print(f"English samples: {english_count}")