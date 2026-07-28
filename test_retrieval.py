"""
test_retrieval.py — Étape 3 : test du pipeline de recherche vectorielle.

Interroge l'index ChromaDB construit par build_index.py et affiche les
passages les plus pertinents pour une question donnée, avec leur score
de similarité et leur provenance (section/sous-section). Sert à valider
la qualité du retrieval AVANT de brancher le LLM (étape 4) : si les bons
passages ne remontent pas ici, aucun réglage du prompt ne rattrapera ça.

Usage :
    python test_retrieval.py
    (puis tape tes questions, une par une ; "quit" pour sortir)
"""

import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. Se reconnecter à l'index existant ---
# Même modèle et même réglage de distance ("cosine") que build_index.py :
# si l'un des deux diffère, les comparaisons de similarité n'ont plus
# de sens.
MODEL_NAME = "intfloat/multilingual-e5-small"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="portfolio_mourad",
    metadata={"hnsw:space": "cosine"},
)

# TOP_K : nombre MAXIMUM de passages retournés par requête (plafond,
# pas un objectif fixe - voir MAX_DISTANCE ci-dessous).
TOP_K = 4

# MAX_DISTANCE : au-delà de ce seuil, un chunk est jugé trop peu
# pertinent pour être gardé, même s'il reste de la place dans TOP_K.
# 0.22 est calibré à partir des observations de test : les bons
# matches se situent autour de 0.13-0.17, donc ce seuil les inclut tous
# largement, tout en excluant les correspondances faibles et génériques
# (ex. un chunk "Contact" qui ne parle d'aucun sujet précis et devient
# donc "moyennement proche" de beaucoup de questions sans être
# franchement pertinent pour aucune).
MAX_DISTANCE = 0.22


def search(query: str, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE):
    """Interroge l'index et affiche les résultats de façon lisible."""
    # Le modèle E5 attend le préfixe "query: " pour une question,
    # différent du préfixe "passage: " utilisé à l'indexation - c'est
    # ce qui permet au modèle de bien différencier "je cherche une
    # info" de "voici une info", et d'améliorer la précision du match.
    query_embedding = model.encode([f"query: {query}"], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    # Avec la distance cosinus, on obtient ici une valeur entre 0 (identique)
    # et 2 (opposé) ; en pratique, un bon match tombe sous ~0.3-0.4.
    distances = results["distances"][0]

    print(f"\n--- Résultats pour : « {query} » ---")
    kept = 0
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        if dist > max_distance:
            # On ne s'arrête pas : les distances sont déjà triées par
            # ordre croissant, donc tout ce qui suit sera encore pire.
            break
        kept += 1
        section = meta.get("section", "?")
        sous_section = meta.get("sous_section", "")
        provenance = f"{section} > {sous_section}" if sous_section else section
        print(f"\n[{i}] distance={dist:.3f} | {provenance}")
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
    if kept == 0:
        print("(aucun résultat suffisamment pertinent)")


if __name__ == "__main__":
    print(f"Index chargé : {collection.count()} chunks disponibles.")
    print("Tape une question (ou 'quit' pour sortir).\n")

    while True:
        query = input("Question > ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        search(query)
