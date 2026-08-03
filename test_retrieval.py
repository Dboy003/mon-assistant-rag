"""
test_retrieval.py — Étape 3 : test du pipeline de recherche.

Interroge l'index (chunks_index.json) construit par build_index.py et
affiche les passages les plus pertinents pour une question donnée, avec
leur score de similarité et leur provenance (section/sous-section).
Sert à valider la qualité du retrieval AVANT de brancher le LLM
(étape 4) : si les bons passages ne remontent pas ici, aucun réglage du
prompt ne rattrapera ça.

Usage :
    python test_retrieval.py
    (puis tape tes questions, une par une ; "quit" pour sortir)
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

# --- 1. Charger l'index existant ---
# Embeddings via l'API Gemini (comme build_index.py et rag.py) plutôt
# qu'un modèle local : voir build_index.py pour l'historique complet
# des tentatives (PyTorch trop lourd, ChromaDB trop lourd, modèle ONNX
# compatible mais peu pertinent). Pas de base de données vectorielle
# non plus : juste un JSON chargé en mémoire et une recherche par
# similarité cosinus en numpy.
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
EMBED_MODEL = "models/gemini-embedding-001"

with open("chunks_index.json", encoding="utf-8") as f:
    _index_data = json.load(f)

_chunk_texts = [c["text"] for c in _index_data]
_chunk_metadatas = [c["metadata"] for c in _index_data]
_chunk_embeddings = np.array([c["embedding"] for c in _index_data], dtype=np.float32)

# TOP_K : nombre MAXIMUM de passages retournés par requête (plafond,
# pas un objectif fixe - voir MAX_DISTANCE ci-dessous).
TOP_K = 6

# MAX_DISTANCE : calibré à partir des observations de test - les bons
# matches se situent autour de 0.26-0.32 avec ce modèle, le bruit
# commence à apparaître à partir de ~0.34.
MAX_DISTANCE = 0.33


def search(query: str, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE):
    """Interroge l'index et affiche les résultats de façon lisible."""
    # task_type="retrieval_query" : même logique que rag.py, signale à
    # l'API que ce texte est une question, pas un document du corpus.
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    query_vector = np.array(result["embedding"], dtype=np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Vecteurs normalisés -> le produit scalaire donne directement la
    # similarité cosinus. On la reconvertit en "distance" (0 = identique).
    similarities = _chunk_embeddings @ query_vector
    distances = 1 - similarities
    top_indices = np.argsort(distances)[:top_k]

    print(f"\n--- Résultats pour : « {query} » ---")
    kept = 0
    for rank, i in enumerate(top_indices, start=1):
        dist = distances[i]
        if dist > max_distance:
            break  # trié par distance croissante : le reste sera pire
        kept += 1
        meta = _chunk_metadatas[i]
        section = meta.get("section", "?")
        sous_section = meta.get("sous_section", "")
        provenance = f"{section} > {sous_section}" if sous_section else section
        print(f"\n[{rank}] distance={dist:.3f} | {provenance}")
        doc = _chunk_texts[i]
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
    if kept == 0:
        print("(aucun résultat suffisamment pertinent)")


if __name__ == "__main__":
    print(f"Index chargé : {len(_chunk_texts)} chunks disponibles.")
    print("Tape une question (ou 'quit' pour sortir).\n")

    while True:
        query = input("Question > ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        search(query)
