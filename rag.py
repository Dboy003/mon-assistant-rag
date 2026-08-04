"""
rag.py — Logique centrale du pipeline RAG (retrieval + génération).

Ce module est le SEUL endroit où vivent le prompt système, le retrieval
et la logique de repli entre modèles. `generate.py` (test en ligne de
commande) et `main.py` (API FastAPI) importent tous les deux `ask()`
depuis ce fichier, pour ne jamais avoir deux versions divergentes du
même comportement.

Configuration :
    Fichier .env (jamais commité) contenant :
        GROQ_API_KEY=...
        GOOGLE_API_KEY=...
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
from langdetect import detect
import numpy as np

# --- 0. Charger les clés API depuis .env ---
load_dotenv()
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- 1. Charger le contexte statique (toujours injecté) ---
# os.path.dirname(__file__) plutôt qu'un chemin relatif nu : garantit
# que ces fichiers sont bien trouvés même si le serveur est lancé
# depuis un autre dossier de travail (cas fréquent une fois déployé,
# par exemple sur Render à l'étape 6).
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "static_context.md"), encoding="utf-8") as f:
    STATIC_CONTEXT = f.read()

# --- 2. Charger l'index (embeddings + textes + métadonnées) ---
# API Gemini (gemini-embedding-001) au lieu d'un modèle local : après
# plusieurs tentatives avec des modèles locaux (voir build_index.py
# pour l'historique complet des essais), calculer les embeddings via
# l'API règle le problème de mémoire à la racine (rien à charger en
# RAM au démarrage du serveur) et restaure une bonne pertinence grâce
# au paramètre task_type, prévu spécifiquement pour la recherche
# question -> passage.
EMBED_MODEL = "models/gemini-embedding-001"

# Pas de base de données vectorielle : pour ~50 chunks, ChromaDB
# (grpcio, kubernetes, opentelemetry...) coûtait plus de RAM à charger
# que tout le reste du pipeline réuni, et faisait dépasser les 512 Mo
# de Render. On charge simplement le JSON produit par build_index.py
# et on cherche par similarité cosinus en numpy - quelques dizaines de
# Ko en mémoire, recherche quasi instantanée à cette échelle.
with open(os.path.join(_here, "chunks_index.json"), encoding="utf-8") as f:
    _index_data = json.load(f)

_chunk_texts = [c["text"] for c in _index_data]
_chunk_metadatas = [c["metadata"] for c in _index_data]
# Une seule matrice (n_chunks, dim) plutôt que n vecteurs séparés :
# permet de calculer toutes les similarités d'un coup (un seul produit
# matriciel) plutôt qu'une boucle Python chunk par chunk.
_chunk_embeddings = np.array([c["embedding"] for c in _index_data], dtype=np.float32)

# Table titre -> texte du chunk correspondant (un projet, une
# expérience...), construite à partir des ### du corpus. Sert à
# l'expansion par référence croisée ci-dessous : un chunk peut citer un
# projet par son nom sans être lui-même le chunk détaillé de ce projet
# (ex. une FAQ qui dit "mon projet préféré est le Biomedical RAG-LLM"
# sans donner les détails techniques, qui vivent dans un autre chunk).
_title_to_chunk = {
    c["metadata"]["sous_section"]: c["text"]
    for c in _index_data
    if c["metadata"].get("sous_section")
}

TOP_K = 6
MAX_DISTANCE = 0.33


def retrieve(question: str) -> str:
    """Récupère les chunks pertinents et les met en forme pour le prompt."""
    # task_type="retrieval_query" : signale à l'API que ce texte est une
    # question de recherche (par opposition à un document du corpus,
    # voir task_type="retrieval_document" dans build_index.py) - c'est
    # ce qui permet à l'API de produire un vecteur bien positionné par
    # rapport aux documents pertinents, même si la question et sa
    # réponse ne se ressemblent pas lexicalement.
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=question,
        task_type="retrieval_query",
    )
    query_vector = np.array(result["embedding"], dtype=np.float32)
    # Normalisation défensive (voir build_index.py pour le raisonnement).
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Les vecteurs étant normalisés, un simple produit scalaire donne
    # directement la similarité cosinus (entre -1 et 1, 1 = identique).
    similarities = _chunk_embeddings @ query_vector
    # On reconvertit en "distance" (0 = identique) pour raisonner avec
    # un seuil intuitif.
    distances = 1 - similarities

    # Indices triés par distance croissante (les plus proches d'abord),
    # puis on ne garde que les TOP_K meilleurs.
    top_indices = np.argsort(distances)[:TOP_K]

    kept = [_chunk_texts[i] for i in top_indices if distances[i] <= MAX_DISTANCE]

    # Expansion par référence croisée : si un chunk gardé mentionne un
    # titre exact (un projet, une expérience) présent dans le corpus,
    # on ajoute aussi CE chunk-là, même si sa propre distance dépasse le
    # seuil. Corrige le cas où seul un chunk "à propos de" (FAQ, résumé)
    # remonte, sans le chunk factuel détaillé qu'il cite.
    for text in list(kept):
        for title, chunk_text in _title_to_chunk.items():
            if title in text and chunk_text not in kept:
                kept.append(chunk_text)

    if not kept:
        return ""
    return "\n\n---\n\n".join(kept)


# --- 3. Prompt système ---
SYSTEM_PROMPT = f"""Tu es l'assistant IA du portfolio de Mourad Do Rego, Data Scientist & GenAI Engineer.
Tu réponds aux visiteurs (recruteurs, professionnels) à la première personne, comme si tu étais Mourad lui-même qui répond à travers cet assistant.

{STATIC_CONTEXT}

Règles impératives :
- Réponds uniquement à partir du contexte fourni ci-dessus et de celui donné dans le message. N'invente et n'extrapole jamais une expérience, une compétence ou un résultat qui n'y figure pas.
- Si le contexte fourni ne permet pas de répondre à la question, dis-le clairement et invite la personne à écrire directement à l'email de contact ci-dessus plutôt que d'improviser une réponse.
- Ton professionnel, courtois et direct en toute circonstance. Pas de familiarité excessive, pas de trait d'humour déplacé.
- Ne prends jamais position sur des sujets politiques, religieux ou polémiques ; recentre poliment sur le profil professionnel.
- IMPORTANT — Langue de réponse : le contexte fourni est presque toujours en français, quelle que soit la langue de la question. Ignore la langue du contexte pour cette décision : réponds TOUJOURS dans la langue de la question posée (traduis mentalement les faits si besoin), jamais dans la langue du contexte par défaut.
- Si la question porte sur une liste ou un panorama (projets réalisés, compétences, expériences), cite plusieurs éléments concrets et nommés présents dans le contexte plutôt qu'un seul exemple développé - privilégie la couverture à la profondeur dans ce cas précis.
- Chaque expérience professionnelle et chaque projet personnel sont des réalisations distinctes, même quand leurs sujets se recoupent (par exemple, une mission professionnelle liée à la fraude et un projet personnel de détection de fraude ne sont pas la même chose). Ne fusionne jamais les détails de deux réalisations différentes en une seule description : garde chaque fait rattaché à sa source précise, identifiable par son titre exact dans le contexte.
- N'établis JAMAIS de lien causal, temporel ou narratif entre deux expériences ou projets distincts (ex. "ce projet découle de cette mission", "en lien avec", "suite à") à moins que ce lien soit écrit explicitement, noir sur blanc, dans le contexte fourni. Une proximité de sujet ou de vocabulaire entre deux chunks n'est PAS une preuve de lien réel : par défaut, traite deux réalisations distinctes comme totalement indépendantes l'une de l'autre.
- Réponses concises : 2 à 5 phrases sauf si la question appelle explicitement un développement plus long ou une liste."""


# --- 4. Génération avec repli automatique à 3 niveaux ---
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=SYSTEM_PROMPT)


def ask(question: str) -> str:
    context = retrieve(question)

    try:
        lang = detect(question)
    except Exception:
        lang = "fr"
    lang_hint = {
        "fr": "Réponds en français.",
        "en": "Answer in English.",
    }.get(lang, "Réponds dans la même langue que la question ci-dessus.")

    user_message = (
        f"Contexte récupéré pour cette question :\n{context}\n\n"
        f"Question du visiteur : {question}\n\n"
        f"[Consigne de langue : {lang_hint}]"
        if context
        else f"(Aucun contexte pertinent trouvé dans la base de connaissances.)\n\n"
             f"Question du visiteur : {question}\n\n"
             f"[Consigne de langue : {lang_hint}]"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Niveau 1 : Llama 3.3 70B (Groq)
    try:
        response = groq_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e1:
        print(f"[Repli sur {FALLBACK_MODEL} (Groq) - raison : {e1}]")

    # Niveau 2 : Llama 3.1 8B (Groq)
    try:
        response = groq_client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e2:
        print(f"[Repli sur {GEMINI_MODEL_NAME} (Google AI) - raison : {e2}]")

    # Niveau 3 : Gemini 2.5 Flash (Google AI)
    try:
        response = gemini_model.generate_content(user_message)
        return response.text
    except Exception as e3:
        return (
            "Désolé, les modèles configurés (Groq et Google AI) sont "
            "actuellement indisponibles. Merci de réessayer plus tard "
            f"ou de me contacter directement par email. (Erreur : {e3})"
        )
