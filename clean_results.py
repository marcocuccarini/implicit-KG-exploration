import json
import os

# 🔹 Directory containing your implicit_ files
INPUT_DIRECTORY = "result"   # Change to your folder path


def extract_steps_from_file(input_path):
    """Extract step-level info from a JSON file"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps_flat = []

    for item in data:
        text_id = item.get("id")
        text = item.get("text")

        for step in item.get("steps", []):
            steps_flat.append({
                "id": text_id,
                "text": text,
                "step": step.get("step"),
                "confidence": step.get("confidence"),
                "explanation": step.get("explanation")
            })

    return steps_flat


def process_implicit_files(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Only process files starting with 'implicit_'
    files = [f for f in os.listdir(directory) if f.endswith(".json") and f.startswith("implicit_")]

    if not files:
        print("No implicit_ JSON files found.")
        return

    for filename in files:
        input_path = os.path.join(directory, filename)

        try:
            steps_flat = extract_steps_from_file(input_path)

            # Save with _steps appended before .json
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_steps.json"
            output_path = os.path.join(directory, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(steps_flat, f, indent=2, ensure_ascii=False)

            print(f"✓ Processed: {filename} → {output_filename}")

        except Exception as e:
            print(f"✗ Failed on {filename}: {str(e)}")


if __name__ == "__main__":
    process_implicit_files(INPUT_DIRECTORY)