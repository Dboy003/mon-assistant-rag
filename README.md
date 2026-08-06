# mon-assistant-rag

Assistant conversationnel RAG (Retrieval-Augmented Generation) qui répond aux questions des visiteurs de mon [portfolio](https://mouradodorego.me), à la première personne, en s'appuyant uniquement sur mon parcours réel : expériences, projets, compétences, aspirations.

**Démo en direct** : [mouradodorego.me](https://mouradodorego.me) → page Assistant IA
**API** : [mon-assistant-rag.onrender.com](https://mon-assistant-rag.onrender.com) · [documentation Swagger](https://mon-assistant-rag.onrender.com/docs)

---

## Pourquoi ce projet

Une page "Contact" classique ne permet pas à un recruteur de creuser une question précise sans m'écrire directement. Cet assistant sert d'interface conversationnelle sur mon portfolio : il peut détailler un projet, expliquer un choix de carrière, ou orienter vers mon email pour tout ce qui dépasse son périmètre, sans jamais inventer une information qui n'est pas dans mon corpus.

## Architecture

```
Question du visiteur
        │
        ▼
┌───────────────────┐
│  Embedding (API    │  gemini-embedding-001, task_type="retrieval_query"
│  Gemini)           │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  Recherche cosinus │  numpy, sur chunks_index.json (pas de base
│  (en mémoire)      │  vectorielle : voir "Pourquoi pas ChromaDB" ci-dessous)
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  Expansion par      │  si un chunk retrouvé cite un projet/expérience
│  référence croisée  │  par son nom, on va aussi chercher son chunk dédié
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  Contexte statique  │  contact, salaire, type de contrat, garde-fous
│  (toujours injecté) │  → jamais soumis au retrieval
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  Génération avec    │  1. Llama 3.3 70B (Groq)
│  repli à 3 niveaux  │  2. Llama 3.1 8B (Groq)
│                     │  3. Gemini 2.5 Flash (Google AI)
└─────────┬──────────┘
          ▼
    Réponse au visiteur
```

## Stack technique

| Composant | Choix | Pourquoi |
|---|---|---|
| Backend | FastAPI + Uvicorn | léger, async, doc Swagger auto-générée |
| Embeddings | API Gemini (`gemini-embedding-001`) | voir la section "Décisions d'ingénierie" : trois autres approches testées et abandonnées avant celle-ci |
| Recherche | numpy (similarité cosinus manuelle) | corpus de ~50 chunks : une vraie base vectorielle est disproportionnée à cette échelle |
| Génération | Groq (Llama 3.3 70B / 3.1 8B) + Gemini 2.5 Flash | repli à 3 niveaux, 2 fournisseurs indépendants |
| Chunking | LangChain (`MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`) | hybride : découpage sémantique par titres, puis sous-découpage à taille fixe si nécessaire |
| Rate limiting | slowapi | 10 req/min, 50 req/jour par IP |
| Déploiement | Render (free tier) | build automatique à chaque push |
| Monitoring | UptimeRobot | ping `/-/healthy` toutes les 5 min, évite la mise en veille |

## Décisions d'ingénierie (et erreurs corrigées en route)

Ce projet a été l'occasion de heurter concrètement les limites du tier gratuit de Render (512 Mo de RAM), et d'itérer jusqu'à une architecture qui tient vraiment dans ce budget :

1. **`sentence-transformers` (PyTorch)** : dépassait 512 Mo dès l'indexation. PyTorch embarque un runtime lourd, même pour un "petit" modèle.
2. **`fastembed` (ONNX) + ChromaDB** : l'indexation passait, mais le *runtime* dépassait encore 512 Mo : ChromaDB embarque des dépendances pensées pour des bases de millions de vecteurs (grpcio, kubernetes, opentelemetry...), disproportionnées pour ~50 chunks.
3. **`fastembed` + recherche cosinus en numpy (sans base de données)** : réglait la mémoire, mais le modèle multilingue compatible (`paraphrase-multilingual-MiniLM-L12-v2`) était moins adapté à la recherche asymétrique question→passage que la famille E5 initialement visée.
4. **Solution retenue : embeddings via l'API Gemini.** Aucun modèle local, mémoire quasi nulle, et un vrai support de la recherche asymétrique via le paramètre `task_type` (`retrieval_query` vs `retrieval_document`), l'équivalent propre et maintenu de ce que faisaient les préfixes `query:`/`passage:` d'E5.

Deux autres réglages fins, découverts en testant avec de vraies questions plutôt qu'en supposant que "ça devrait marcher" :

- **Expansion par référence croisée** : un chunk FAQ qui dit *"mon projet préféré est le Biomedical RAG-LLM"* remonte bien pour une question sur ce sujet, mais le chunk technique détaillé du projet, lui, ne matche pas assez bien lexicalement pour apparaître dans le top-k. Solution : si un chunk retenu cite un titre exact du corpus (projet, expérience), son chunk dédié est automatiquement ajouté au contexte, même si sa distance dépasse le seuil.
- **Contexte statique séparé du corpus recherchable** : le contact, les prétentions salariales et les garde-fous de comportement ne sont *jamais* indexés pour la recherche. Ce sont des chunks très courts et génériques qui devenaient des "attracteurs de bruit" (moyennement proches de beaucoup de questions sans être vraiment pertinents pour aucune). Ils sont à la place injectés systématiquement dans le prompt système, garantissant leur disponibilité indépendamment de la question posée.

## Garde-fous

Le prompt système impose plusieurs règles strictes, notamment :
- Ne jamais répondre en dehors du contexte fourni (pas d'invention).
- Ne jamais établir de lien causal entre deux expériences/projets distincts, sauf si ce lien est écrit noir sur blanc dans le corpus.
- Toujours répondre dans la langue de la question, indépendamment de la langue du contexte (le corpus est en français).
- Rediriger vers l'email de contact pour toute question hors périmètre.

## Structure du projet

```
mon-assistant-rag/
├── app.py                  # API FastAPI (endpoints /, /-/healthy, /chat)
├── rag.py                  # Logique métier centrale (retrieval + génération) - seul endroit où elle vit
├── build_index.py          # Construction de l'index (chunking + embeddings) à partir de knowledge_base.md
├── generate.py             # Script CLI pour tester le pipeline en local
├── test_retrieval.py       # Script CLI pour déboguer uniquement le retrieval
├── knowledge_base.md       # Corpus source, structuré en ## (page) et ### (unité sémantique)
├── static_context.md       # Contexte toujours injecté (contact, salaire, garde-fous)
├── chunks_index.json       # Index généré (embeddings + textes), non commité, régénéré au build
└── requirements.txt
```

## Lancer en local

```bash
git clone https://github.com/Dboy003/mon-assistant-rag.git
cd mon-assistant-rag
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Créer un fichier `.env` :
```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

Construire l'index puis lancer l'API :
```bash
python build_index.py
uvicorn app:app --reload --port 8000
```

Documentation interactive sur `http://localhost:8000/docs`.

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Vérification manuelle rapide |
| `GET`/`HEAD` | `/-/healthy` | Endpoint de santé (monitoring UptimeRobot) |
| `POST` | `/chat` | `{"message": "..."}` → `{"reply": "..."}` |

## Déploiement

Hébergé sur **Render** (tier gratuit) :
- **Build Command** : `pip install -r requirements.txt && python build_index.py`
- **Start Command** : `uvicorn app:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'`
- Variables d'environnement : `GROQ_API_KEY`, `GOOGLE_API_KEY`

---

## Projet associé

Ce backend sert le [portfolio](https://github.com/Dboy003/portfolio-mourad) sur lequel il est intégré. Même architecture RAG (LangChain + Llama 3.3 70B via Groq) que mon projet [Biomedical RAG-LLM](https://github.com/Dboy003/biomedical-rag-llm), adaptée ici à un corpus personnel plutôt qu'à la littérature scientifique PubMed.

## Auteur

**Mourad Do Rego** · Data Scientist & GenAI Engineer
[Portfolio](https://mouradodorego.me) · [GitHub](https://github.com/Dboy003) · mouwahiddorego@gmail.com
