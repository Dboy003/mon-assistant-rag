"""
generate.py — Script CLI pour tester le pipeline RAG en local.

Toute la logique (retrieval, prompt système, repli entre modèles) vit
dans rag.py : ce fichier ne fait qu'appeler ask() dans une boucle
interactive, pour tester rapidement sans passer par l'API.

Usage :
    python generate.py
"""

from rag import ask

if __name__ == "__main__":
    print("Assistant RAG prêt. Tape une question (ou 'quit' pour sortir).\n")
    while True:
        q = input("Question > ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        print("\n" + ask(q) + "\n")
