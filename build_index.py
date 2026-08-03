"""
build_index.py — Étape 2 : chunking hybride + embeddings du corpus du portfolio.

Construit l'index à partir de knowledge_base.md :
1. Découpage par titres Markdown (## et ###) pour préserver la cohérence sémantique.
2. Sous-découpage à taille fixe (avec chevauchement) uniquement pour les
   sections qui dépassent la taille cible.
3. Génération des embeddings via l'API Gemini (gemini-embedding-001).
4. Sauvegarde dans un simple fichier JSON (chunks_index.json).

Pourquoi pas ChromaDB : conçu pour des bases de milliers/millions de
vecteurs, ses dépendances (grpcio, kubernetes, opentelemetry...) sont
disproportionnées pour un corpus de moins de 100 chunks et faisaient
dépasser les 512 Mo de RAM du tier gratuit de Render. Pour ce volume,
une recherche par similarité cosinus "à la main" en numpy est aussi
rapide et ne pèse presque rien en mémoire.

Pourquoi l'API plutôt qu'un modèle local : après plusieurs tentatives
avec des modèles locaux (PyTorch, puis ONNX) toujours trop lourds ou
pas assez pertinents pour la recherche question->passage, calculer les
embeddings via l'API Gemini règle le problème de mémoire à la racine
(rien à charger en RAM) et restaure une bonne pertinence grâce au
paramètre task_type, prévu spécifiquement pour ce cas d'usage.

Installation :
    pip install langchain-text-splitters google-generativeai numpy python-dotenv --break-system-packages

Configuration :
    Fichier .env contenant GOOGLE_API_KEY=ta_clé_ici

Usage :
    python build_index.py
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import json
import os

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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

# --- 3bis. Garde-fou : valeur de repli pour les métadonnées vides ---
# Le texte situé avant le tout premier "##" (le titre du document et sa
# description) n'a ni "section" ni "sous_section" associée.
for doc in final_chunks:
    if not doc.metadata:
        doc.metadata = {"section": "Introduction"}

# --- 4. Génération des embeddings ---
# API Gemini (gemini-embedding-001) au lieu d'un modèle local : après
# 3 tentatives côté modèle local (PyTorch trop lourd, ChromaDB trop
# lourd, modèle compatible ONNX mais moins pertinent pour la recherche
# question->passage), on calcule les embeddings via l'API plutôt qu'en
# local. Plus aucun modèle à télécharger ni à charger en mémoire sur
# Render - le principal compromis est que Google AI devient une
# dépendance nécessaire pour la recherche elle-même (plus seulement un
# filet de sécurité de niveau 3 comme pour la génération).
# task_type="retrieval_document" : indique explicitement à l'API que ce
# texte fait partie du corpus à retrouver (par opposition à une
# question) - l'équivalent propre des préfixes "query:"/"passage:" de
# la famille E5, mais géré nativement par l'API plutôt que par convention.
EMBED_MODEL = "models/gemini-embedding-001"

passage_texts = [c.page_content for c in final_chunks]

result = genai.embed_content(
    model=EMBED_MODEL,
    content=passage_texts,
    task_type="retrieval_document",
)

# Normalisation manuelle défensive : même si l'API renvoie déjà des
# vecteurs normalisés dans la plupart des cas, on ne suppose plus rien
# après s'être fait surprendre deux fois par cette hypothèse - la
# division par la norme L2 est sans risque même si c'était déjà fait.
def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).tolist()

embeddings = [normalize(vec) for vec in result["embedding"]]

# --- 5. Sauvegarde dans un fichier JSON unique ---
# Un seul fichier plat, sans base de données : { texte, métadonnées,
# vecteur } pour chaque chunk. Chargé intégralement en mémoire au
# démarrage du serveur (quelques dizaines de Ko pour ce corpus), puis
# parcouru par similarité cosinus - voir rag.py.
index_data = [
    {
        "text": c.page_content,
        "metadata": c.metadata,
        "embedding": emb,
    }
    for c, emb in zip(final_chunks, embeddings)
]

with open("chunks_index.json", "w", encoding="utf-8") as f:
    json.dump(index_data, f, ensure_ascii=False)

print(f"Index créé dans chunks_index.json avec {len(index_data)} chunks.")
