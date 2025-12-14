# 💻 Code de l'Application

Ce dossier contient le code source de l'application Streamlit et les modules de backend pour l'analyse RAG et le modèle de sentiment.

## 📂 Contenu

### Interface Utilisateur
- **[MacroLLM_app.py](MacroLLM_app.py)** : Point d'entrée de l'application Streamlit.
    - Configure l'interface utilisateur (Dashboard financier).
    - Gère les interactions utilisateur (sélection de devise, événement, etc.).
    - Affiche les résultats de l'analyse (Sentiment, RAG, Recommandation Gemini).

### Backend & Pipeline
- **[pipeline_RAG.py](pipeline_RAG.py)** : Orchestre le pipeline RAG (Retrieval-Augmented Generation).
    - Initialise le modèle **Google Gemini**.
    - Charge le dataset avec les embeddings.
    - Combine l'analyse de sentiment et la recherche de similarité pour générer un prompt pour le LLM.
    - Fonction principale : `generate_analysis_v2`.

- **[sentiment_analysis.py](sentiment_analysis.py)** : Module d'analyse de sentiment.
    - Charge le modèle **DistilRoBERTa fine-tuné**.
    - Prédit l'impact de l'annonce (Hausse/Baisse/Neutre).
    - Fonction principale : `get_sentiment_distilroberta`.

- **[similarity_search.py](similarity_search.py)** : Module de recherche vectorielle.
    - Charge le modèle d'embeddings (DistilRoBERTa).
    - Calcule les embeddings pour les nouvelles annonces.
    - Effectue la recherche de similarité cosinus pour trouver des événements historiques similaires.
    - Fonction principale : `find_similar_events_v2`.

## 🚀 Comment lancer l'application

Pour lancer l'application, exécutez la commande suivante depuis la racine du projet :

```bash
streamlit run app_interface_code/MacroLLM_app.py
```

Assurez-vous d'avoir installé les dépendances nécessaires et que le modèle fine-tuné est présent dans le dossier racine (dossier `distilroberta_forex_final`).
