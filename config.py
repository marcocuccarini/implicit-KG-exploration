TOP_N_PER_SOURCE = 15
import re
STER_URI = "http://example.org/ster#"
LLM_MODEL = "gemma3:4b"
DATASET_PATH = "data/data/dataset_split_test.csv"
safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', LLM_MODEL)  # replaces : or . with _
RESULTS_PATH = f"result/implicit_results_{safe_model_name}.json"
CACHE_FILE = "wikidata_cache.json"
LOCAL_KG_PATH = "kg/output_final.ttl"

