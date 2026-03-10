import pandas as pd

def merge_csv_json(csv_file, json_file, prefix):
    # load files
    df_csv = pd.read_csv(csv_file)
    df_json = pd.read_json(json_file)

    # rename json columns
    df_json = df_json.rename(columns={
        "explanation": f"baseline_target",
        "confidence": f"baseline_target_confidence",
    })

    # merge on id
    merged = df_csv.merge(df_json, on="id", how="left")

    return merged


gemma = merge_csv_json("implicit_results_gpt-oss_20b_auto_eval_tiebreak_modes.csv", "baseline_gpt-oss_20b.json", "gemma")
got = merge_csv_json("implicit_results_gemma3_4b_auto_eval_tiebreak_modes.csv", "baseline_gemma3_4b.json", "got")
qwen = merge_csv_json("implicit_results_qwen3_8b_auto_eval_tiebreak_modes.csv", "baseline_qwen3_8b.json", "qwen")

# save results
gemma.to_csv("gemma3_merged.csv", index=False)
got.to_csv("got-oss_merged.csv", index=False)
qwen.to_csv("qwen3_merged.csv", index=False)