from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticTripleFilter:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def filter_triples(self, text, triples, top_k=15):
        if not triples:
            return []

        # Prepare triple texts
        triple_texts = [
            f"{t['subject']} {t['predicate']} {t['object']}"
            for t in triples
        ]

        # Embed text and triples
        text_embedding = self.embed([text])
        triple_embeddings = self.embed(triple_texts)

        # Compute cosine similarity
        similarities = cosine_similarity(text_embedding, triple_embeddings)[0]

        # Attach score
        for t, score in zip(triples, similarities):
            t["semantic_score"] = float(score)

        # Sort by similarity
        ranked = sorted(triples, key=lambda x: x["semantic_score"], reverse=True)

        return ranked[:top_k]