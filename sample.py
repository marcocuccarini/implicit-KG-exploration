import json
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("result")
N = 2000

json_files = list(INPUT_DIR.glob("*.json"))

for json_file in json_files:

    print(f"Processing {json_file.name}")

    # Load JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by id
    data_sorted = sorted(data, key=lambda x: x["id"])

    # Select first 2000 samples
    subset = data_sorted[:N]

    # Overwrite file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"{json_file.name} → kept {len(subset)} samples")

print("\nAll files processed.")