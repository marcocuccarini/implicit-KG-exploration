import json
from collections import defaultdict
from utils.json_utils import safe_json_load

# --- CONFIGURATION ---
CONF_THRESHOLD = 0.90
MAX_STEPS = 4
MAX_NEXT_NODES = 2
LLM_CACHE = {}

# --- PROMPTS ---

def implicit_explanation_prompt(text):
    return f"""
You are an expert in pragmatics and stereotypes.
TEXT: \"\"\"{text}\"\"\"
TASK: Explain the implicit meaning and bias conveyed in the text. 
Estimate your confidence in this explanation.
Return ONLY valid JSON:
{{ "explanation": "string", "confidence": float }}
"""

def combined_reasoning_prompt(text, triples):
    triples_str = "\n".join(
        [f"{i}: {t['subject']} | {t['predicate']} | {t['object']} ({t['source']})"
         for i, t in enumerate(triples)]
    )

    return f"""
ANALYSIS ROLE: Strict Sociolinguistic Auditor

TEXT:
"{text}"

TRIPLES:
{triples_str}

TASK:
1. Select indices of triples that explicitly document a stereotype or cultural association.
2. Explain the bias ONLY using the provided TRIPLES as evidence.
3. STRICT CONFIDENCE SCORING (0.0 - 1.0):
   - 0.90+: Explicit link found.
   - PENALTY: Reduce score by 0.3 if relying on internal knowledge.
4. If confidence < 0.95, select TWO entities to explore further.

Return ONLY JSON:
{{
  "relevant_indices": [int],
  "explanation": "string",
  "confidence": float,
  "next_nodes": [
    {{"wikidata": "formal_name", "conceptnet": "keyword"}},
    {{"wikidata": "formal_name", "conceptnet": "keyword"}}
  ]
}}
"""

def filtering_context_prompt(text, triples):
    """Phase 1: Broad Filter for cultural, social, or associative context."""
    triples_str = "\n".join([
        f"{i}: {t['subject']} | {t['predicate']} | {t['object']} ({t['source']})"
        for i, t in enumerate(triples)
    ])
    
    return f"""
ROLE: Stereotype & Context Auditor (Broad Search)
TEXT: "{text}"

TASK:
Identify any triples that could help explain the subtext, bias, or imagery in the text.
Be inclusive. Select a triple if it meets ANY of these criteria:
1. DIRECT CONTEXT: Historical roles, symbols, or stereotypes.
2. INDIRECT ASSOCIATION: Concepts that are generally related to the targets or the bias.
3. LINGUISTIC LINKS: Synonyms or "is-a" relationships that clarify what the entities are.
4. COMMONSENSE: General knowledge that provides a "bridge" to understand the text.

Return ONLY JSON:
{{ "relevant_indices": [int, int] }}

CANDIDATE TRIPLES:
{triples_str}
"""

def explanation_reasoning_prompt(text, filtered_triples):
    triples_str = "\n".join(
        [f"- {t['subject']} {t['predicate']} {t['object']} (Source: {t['source']})"
         for t in filtered_triples]
    )

    return f"""
ROLE: Sociolinguistic Analyst
TEXT: "{text}"
EVIDENCE:
{triples_str}

TASK:
1. Explain the implicit bias/stereotype in the text using ONLY the provided EVIDENCE.
2. Provide a confidence score (0.0 - 1.0). 
   - 0.90+ requires explicit grounding in the triples.
   - Penalty (-0.3) if you use external knowledge not in the evidence.
3. Suggest two specific entities (nodes) to explore next if more info is needed.

Return ONLY JSON:
{{
  "explanation": "string",
  "confidence": float,
  "next_nodes": [
    {{"wikidata": "Entity1", "conceptnet": "key1"}},
    {{"wikidata": "Entity2", "conceptnet": "key2"}}
  ]
}}
"""

# --- HELPER FUNCTIONS ---
def top_n_per_source(triples, n_per_source=5):
    from collections import defaultdict

    # Group triples by source
    source_dict = defaultdict(list)
    for t in triples:
        source_dict[t.get("source", "unknown")].append(t)

    # Take top N per source based on semantic_score
    final_triples = []
    for source, ts in source_dict.items():
        # Sort descending by semantic_score
        ranked = sorted(ts, key=lambda x: x.get("semantic_score", 0), reverse=True)
        final_triples.extend(ranked[:n_per_source])

    return final_triples

def cached_llm_call(llm, prompt):
    global LLM_CACHE
    if prompt in LLM_CACHE:
        return LLM_CACHE[prompt]
    
    # Assuming your LLM class has a send_prompt method
    response = llm.send_prompt(prompt) 
    LLM_CACHE[prompt] = response
    return response

# --- MAIN FUNCTION ---

def iterative_explanation(text_id, text, targets, llm, explorer, sources, semantic_filter):
    report = {
        "id": text_id,
        "text": text,
        "target": targets,
        "final_confidence": 0.0,
        "steps": []
    }
    
    visited_nodes = set([t.lower().strip() for t in targets])
    current_search_packets = [{"wikidata": t, "conceptnet": t, "local": t} for t in targets]

    # --- STEP 0: SKEPTICAL HYPOTHESIS ---
    raw0 = cached_llm_call(llm, implicit_explanation_prompt(text))
    res0 = safe_json_load(raw0) or {}
    conf0 = min(res0.get("confidence", 0.0), 0.85) 
    
    print(f"\n[STEP 0] Initial Hypothesis Confidence: {conf0}")
    
    report["steps"].append({
        "step": 0,
        "explanation": res0.get("explanation", ""),
        "confidence": conf0,
        "relevant_triples": [],
        "reached_threshold": False
    })

    # --- MULTI-HOP GROUNDING ---
    for step_id in range(1, MAX_STEPS + 1):
        all_step_triples = []
        nodes_this_step = [p.get("local", p.get("wikidata")) for p in current_search_packets]
        
        for packet in current_search_packets:
            arch_json = {"target_map": packet, "context_entity": {}}
            triples = explorer.get_triples_for_architect_query(arch_json, sources)
            all_step_triples.extend(triples)

        if not all_step_triples:
            print(f"[STEP {step_id}] No triples found for: {nodes_this_step}")
            break

        # --- PHASE 1: FILTERING FOR CONTEXT ---
        print(f"\n--- [STEP {step_id}: PHASE 1 - FILTERING] ---")
        print(f"Total Triples to evaluate: {len(all_step_triples)}")
        
        filter_raw = cached_llm_call(llm, filtering_context_prompt(text, all_step_triples))
        filter_res = safe_json_load(filter_raw) or {}
        indices = filter_res.get("relevant_indices", [])
        
        relevant_triples = [
            all_step_triples[i] for i in indices 
            if isinstance(i, int) and i < len(all_step_triples)
        ]

        # PRINT SELECTED TRIPLES
        if relevant_triples:
            print(f"Auditor SELECTED {len(relevant_triples)} contextual triples:")
            for rt in relevant_triples:
                print(f"  > [SELECTED]: {rt.get('subject')} --({rt.get('predicate')})--> {rt.get('object')} [{rt.get('source')}]")
        else:
            print("Auditor rejected all candidate triples as irrelevant.")

        # --- PHASE 2: REASONING & EXPLANATION ---
        print(f"\n--- [STEP {step_id}: PHASE 2 - REASONING] ---")
        reasoning_raw = cached_llm_call(llm, explanation_reasoning_prompt(text, relevant_triples))
        res_step = safe_json_load(reasoning_raw) or {}
        
        raw_conf = res_step.get("confidence", 0.0)
        current_conf = 0.3 if (not relevant_triples and raw_conf > 0.5) else raw_conf

        print(f"Explanation: {res_step.get('explanation', 'No explanation generated.')}")
        print(f"Confidence: {current_conf} (Evidence Grounding: {'YES' if relevant_triples else 'NO'})")

        step_record = {
            "step": step_id,
            "explanation": res_step.get("explanation", ""),
            "confidence": current_conf,
            "nodes_queried": nodes_this_step,
            "relevant_triples": relevant_triples,
            "reached_threshold": current_conf >= CONF_THRESHOLD
        }
        report["steps"].append(step_record)
        report["final_confidence"] = current_conf

        if current_conf >= CONF_THRESHOLD:
            print(f"Success: Threshold {CONF_THRESHOLD} reached.")
            break

        # Prepare Next Hop
        new_packets = []
        for node in res_step.get("next_nodes", []):
            node_id = node.get("wikidata", "").lower().strip()
            if node_id and node_id not in visited_nodes:
                new_packets.append({
                    "wikidata": node["wikidata"], 
                    "conceptnet": node["conceptnet"], 
                    "local": node["wikidata"] 
                })
                visited_nodes.add(node_id)
        
        if not new_packets: break
        print(f"Expanding search to nodes: {[n['wikidata'] for n in new_packets]}")
        current_search_packets = new_packets[:MAX_NEXT_NODES] 

    return report

# --- NEW PROMPT FUNCTIONS ---

def filtering_context_prompt(text, triples):
    """Phase 1: Filter triples that provide cultural/social context."""
    triples_str = "\n".join([
        f"{i}: {t['subject']} | {t['predicate']} | {t['object']} ({t['source']})"
        for i, t in enumerate(triples)
    ])
    return f"""
ROLE: Stereotype & Context Auditor
TEXT: "{text}"

TASK:
From the list below, select only the triples that provide cultural, historical, or social CONTEXT to understand the stereotype in the text.
Look for:
- Historical roles or symbolic associations.
- Common cultural tropes (e.g., "is-a", "has-property").
- Traditional occupations or traits linked to the targets.

Return ONLY JSON:
{{ "relevant_indices": [int, int] }}

CANDIDATE TRIPLES:
{triples_str}
"""

def explanation_reasoning_prompt(text, context_triples):
    """Phase 2: Generate explanation based ONLY on filtered context."""
    triples_str = "\n".join([
        f"- {t['subject']} {t['predicate']} {t['object']} (Source: {t['source']})"
        for t in context_triples
    ])
    return f"""
ROLE: Sociolinguistic Analyst
TEXT: "{text}"
SELECTED CONTEXTUAL EVIDENCE:
{triples_str}

TASK:
1. Using ONLY the provided context, explain the implicit bias or stereotype in the text.
2. Provide a confidence score (0.0 - 1.0). 
3. If more info is needed, suggest two nodes for the next hop.

Return ONLY JSON:
{{
  "explanation": "string",
  "confidence": float,
  "next_nodes": [
    {{"wikidata": "name", "conceptnet": "key"}},
    {{"wikidata": "name", "conceptnet": "key"}}
  ]
}}
"""