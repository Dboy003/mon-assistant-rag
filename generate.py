"""
Pipeline complet : question -> retrieval (ChromaDB + e5-small) -> prompt
(contexte statique + chunks récupérés) -> génération (Llama 3.3 70B via
Groq, avec repli automatique sur Llama 3.1 8B).

"""

import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
from langdetect import detect
import chromadb
from sentence_transformers import SentenceTransformer

# --- 0. Charger les clés API depuis .env ---
load_dotenv()
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- 1. Charger le contexte statique (toujours injecté) ---
with open("static_context.md", encoding="utf-8") as f:
    STATIC_CONTEXT = f.read()

# --- 2. Se reconnecter à l'index ChromaDB (identique à test_retrieval.py) ---
MODEL_NAME = "intfloat/multilingual-e5-small"
embed_model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="portfolio_mourad",
    metadata={"hnsw:space": "cosine"},
)

TOP_K = 6
MAX_DISTANCE = 0.22


def retrieve(question: str) -> str:
    """Récupère les chunks pertinents et les met en forme pour le prompt."""
    query_embedding = embed_model.encode(
        [f"query: {question}"], normalize_embeddings=True
    ).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

    documents = results["documents"][0]
    distances = results["distances"][0]

    kept = [doc for doc, dist in zip(documents, distances) if dist <= MAX_DISTANCE]
    if not kept:
        return ""
    return "\n\n---\n\n".join(kept)


# --- 3. Prompt système ---
# Structure : (a) rôle et ton, (b) contexte statique (toujours présent,
# jamais soumis au retrieval), (c) garde-fous de comportement,
# (d) consigne de langue.
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
# Niveau 1 : Llama 3.3 70B (Groq) - meilleure qualité.
# Niveau 2 : Llama 3.1 8B (Groq) - même fournisseur, quota différent ;
#            couvre le cas où seul le modèle 70B est temporairement
#            indisponible ou en rate-limit.
# Niveau 3 : Gemini 2.5 Flash (Google AI) - fournisseur totalement
#            différent de Groq ; couvre le cas où c'est Groq lui-même
#            (les deux modèles à la fois) qui est en panne ou en
#            rate-limit, ce qu'aucun repli intra-Groq ne peut couvrir.
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Instancié une seule fois (pas à chaque question) : le system_instruction
# est fixe, pas besoin de recréer l'objet à chaque appel.
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=SYSTEM_PROMPT)


def ask(question: str) -> str:
    context = retrieve(question)

    # On détecte la langue de la question et on répète la consigne de
    # langue juste à côté d'elle, dans le message utilisateur - pas
    # seulement dans le prompt système. Un LLM accorde généralement
    # plus de poids aux instructions proches de la tâche à accomplir
    # qu'à une règle noyée plus haut dans un long prompt système,
    # surtout quand le contexte fourni est volumineux et dans une
    # autre langue (ce qui était la cause du bug observé : la question
    # en anglais recevait une réponse en français).
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
            temperature=0.2,  # basse : on veut de la fidélité au contexte, pas de créativité (baissée de 0.3 à 0.2 après un cas d'invention de lien entre deux réalisations)
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e1:
        print(f"[Repli sur {FALLBACK_MODEL} (Groq) - raison : {e1}]")

    # Niveau 2 : Llama 3.1 8B (Groq) - même fournisseur, quota séparé
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

    # Niveau 3 : Gemini 2.5 Flash (Google AI) - fournisseur différent,
    # pour couvrir le cas où Groq est indisponible dans son ensemble.
    # L'API Gemini reçoit le system_instruction séparément (configuré
    # une fois, voir gemini_model plus haut) : on ne lui passe donc que
    # le message utilisateur, pas la liste "messages" au format Groq.
    try:
        response = gemini_model.generate_content(user_message)
        return response.text
    except Exception as e3:
        return (
            "Désolé, les modèles configurés (Groq et Google AI) sont "
            "actuellement indisponibles. Merci de réessayer plus tard "
            f"ou de me contacter directement par email. (Erreur : {e3})"
        )


if __name__ == "__main__":
    print("Assistant RAG prêt. Tape une question (ou 'quit' pour sortir).\n")
    while True:
        q = input("Question > ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        print("\n" + ask(q) + "\n")
