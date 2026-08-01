"""
build_index.py — Étape 2 : chunking hybride + embeddings du corpus du portfolio.

Construit l'index vectoriel ChromaDB à partir de knowledge_base.md :
1. Découpage par titres Markdown (## et ###) pour préserver la cohérence sémantique.
2. Sous-découpage à taille fixe (avec chevauchement) uniquement pour les
   sections qui dépassent la taille cible.
3. Génération des embeddings (all-MiniLM-L6-v2, comme le projet Biomedical RAG).
4. Indexation dans une collection ChromaDB persistante locale.

Installation :
    pip install langchain-text-splitters chromadb sentence-transformers --break-system-packages

Usage :
    python build_index.py
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import chromadb

# --- 1. Charger le corpus ---
# knowledge_base.md doit être dans le même dossier que ce script.
with open("knowledge_base.md", encoding="utf-8") as f:
    raw_text = f.read()

# --- 2. Découpage par titres (## et ###) ---
# On demande au splitter de repérer les niveaux de titre et de les
# conserver comme métadonnées ("section", "sous_section") sur chaque
# chunk : ça permettra plus tard de savoir de quelle partie du site
# vient un passage retrouvé (utile pour le debug et pour d'éventuels
# filtres de recherche par section).
headers_to_split_on = [
    ("##", "section"),
    ("###", "sous_section"),
]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
header_chunks = header_splitter.split_text(raw_text)

# --- 3. Sous-découpage des sections trop longues ---
# TARGET_SIZE : au-delà de ~800 caractères, un embedding devient moins
# précis car il doit "résumer" trop d'idées différentes en un seul vecteur.
# OVERLAP : 100 caractères de recouvrement entre deux chunks voisins,
# pour ne pas couper une idée pile à la frontière (ex. séparer une
# statistique de sa légende).
# La liste `separators` dit au splitter d'essayer de couper d'abord aux
# paragraphes, puis aux puces, puis aux phrases, et seulement en dernier
# recours au milieu d'une phrase.
TARGET_SIZE = 800
OVERLAP = 100
sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TARGET_SIZE,
    chunk_overlap=OVERLAP,
    separators=["\n\n", "\n- ", "\n", ". ", " "],
)

final_chunks = []
for doc in header_chunks:
    if len(doc.page_content) <= TARGET_SIZE:
        # Section déjà assez courte : on la garde telle quelle,
        # pas besoin de la découper davantage.
        final_chunks.append(doc)
    else:
        # Section trop longue (ex. le détail du projet E-commerce
        # Pricing) : on la sous-découpe, en gardant les métadonnées
        # ("section"/"sous_section") sur chaque sous-chunk.
        sub_docs = sub_splitter.split_documents([doc])
        final_chunks.extend(sub_docs)

print(f"{len(header_chunks)} sections détectées -> {len(final_chunks)} chunks finaux")

# --- 3bis. Garde-fou : ChromaDB refuse un dict de métadonnées vide ---
# Le texte situé avant le tout premier "##" (le titre du document et sa
# description) n'a ni "section" ni "sous_section" associée, donc son
# dictionnaire de métadonnées est vide. On lui donne une valeur de
# repli plutôt que de le laisser planter l'indexation.
for doc in final_chunks:
    if not doc.metadata:
        doc.metadata = {"section": "Introduction"}

# --- 4. Génération des embeddings + indexation ChromaDB ---
# Modèle multilingual supporté par fastembed.
# On conserve la convention E5 query/passage pour rester cohérent avec
# la logique de retrieval du module rag.py.
MODEL_NAME = "intfloat/multilingual-e5-large"
model = TextEmbedding(model_name=MODEL_NAME)

# On préfixe une copie du texte pour le calcul de l'embedding, mais on
# garde le texte original (sans préfixe) pour l'affichage des résultats.
# model.embed() renvoie un générateur (un vecteur numpy par texte en
# entrée, déjà normalisé) : on le convertit en liste de listes, le
# format attendu par ChromaDB.
passage_texts = [f"passage: {c.page_content}" for c in final_chunks]
embeddings = [vec.tolist() for vec in model.embed(passage_texts)]

# PersistentClient écrit l'index sur disque (dossier ./chroma_db) :
# contrairement à un client en mémoire, l'index survit au redémarrage
# du script, donc on ne recalcule pas les embeddings à chaque fois.
client = chromadb.PersistentClient(path="./chroma_db")

# On supprime la collection existante avant de la recréer : sans ça,
# relancer ce script plusieurs fois (par ex. après avoir modifié
# knowledge_base.md) accumule les anciens ET les nouveaux chunks côte
# à côte, avec des ID en collision que ChromaDB ignore silencieusement
# au lieu de les mettre à jour. Ce script doit rester idempotent :
# le relancer doit toujours donner un index propre, reflétant
# uniquement le contenu actuel de knowledge_base.md.
try:
    client.delete_collection(name="portfolio_mourad")
except Exception:
    pass  # la collection n'existe pas encore lors du tout premier lancement

# metadata={"hnsw:space": "cosine"} : les embeddings E5 sont conçus
# pour être comparés par similarité cosinus (d'où le
# normalize_embeddings=True ci-dessus) plutôt que par distance
# euclidienne, qui est le réglage par défaut de ChromaDB.
collection = client.get_or_create_collection(
    name="portfolio_mourad",
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(final_chunks))],
    documents=[c.page_content for c in final_chunks],
    embeddings=embeddings,
    metadatas=[c.metadata for c in final_chunks],
)

print(f"Index ChromaDB créé dans ./chroma_db avec {collection.count()} chunks.")
