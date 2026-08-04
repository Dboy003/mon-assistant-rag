"""
app.py — Étape 5 : exposer le pipeline RAG via une API FastAPI.

Réutilise directement rag.ask() (retrieval + génération avec repli à
3 niveaux) : ce fichier n'est qu'une fine couche HTTP par-dessus un
pipeline déjà testé, pas une réécriture.

Installation :
    pip install fastapi "uvicorn[standard]"

Lancement en local :
    uvicorn app:app --reload --port 8000

Test rapide (dans un autre terminal) :
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Parle-moi de Carrefour\"}"
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import ask  # importer rag.py charge les modèles UNE SEULE FOIS
                      # au démarrage du serveur (pas à chaque requête) :
                      # c'est tout l'intérêt de passer d'un script CLI
                      # à un serveur qui tourne en continu.

app = FastAPI(
    title="Assistant RAG - Portfolio Mourad Do Rego",
    description="API du chatbot RAG interrogeant le corpus du portfolio.",
    version="1.0.0",
)

# CORS : par défaut, un navigateur bloque les appels JS depuis
# GitHub Pages (dboy003.github.io) vers une API sur un autre domaine
# (Render), sauf si le serveur autorise explicitement cette origine.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dboy003.github.io",
        "https://mouradodorego.me",
        "https://www.mouradodorego.me",
        "http://mouradodorego.me",  # au cas où le HTTPS n'est pas encore pleinement provisionné
        "http://localhost:8000",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    """Simple point de contrôle manuel : pratique pour vérifier que
    l'API tourne en ouvrant juste l'URL dans un navigateur."""
    return {"status": "ok", "service": "portfolio-rag-assistant"}


@app.get("/-/healthy")
def health():
    """Endpoint de santé dédié au monitoring (UptimeRobot), même
    convention que sur le projet Fraud Detection Banking."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        reply = ask(req.message)
    except Exception as e:
        # ask() gère déjà en interne l'échec des 3 modèles (Groq x2 +
        # Gemini) sans lever d'exception - si on arrive quand même ici,
        # c'est un problème plus profond (retrieval, index absent...).
        # On renvoie une erreur HTTP propre plutôt qu'un stack trace
        # brut au front-end.
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")
    return ChatResponse(reply=reply)
