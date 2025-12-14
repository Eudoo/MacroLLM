# MacroLLM - Assistant Forex IA

**MacroLLM** est un assistant d'analyse Forex basé sur l'IA qui combine un modèle **DistilRoBERTa fine-tuné** et une architecture **RAG (Retrieval-Augmented Generation)** pour interpréter les événements macroéconomiques et générer des insights de trading via **Google Gemini**.

## 🎯 Objectifs
- Prédire l'impact des annonces économiques sur les paires de devises.
- Fournir des analyses contextuelles basées sur l'historique du marché.
- Générer des recommandations de trading actionnables.

---

## 🏗️ Architecture du Projet

Le projet est structuré en plusieurs modules. Cliquez sur les liens pour accéder à la documentation détaillée de chaque partie.

```
MacroLLM/
├── 📄 README.md (Ce fichier)
├── 📄 requirements.txt .............. Dépendances Python
├── 📂 app_interface_code/ ........... Code source de l'application et du pipeline
│   ├── 📄 README.md ................. Documentation des scripts Python
│   ├── 🐍 MacroLLM_app.py ........... Interface Streamlit
│   ├── 🐍 pipeline_RAG.py ........... Logique RAG et intégration Gemini
│   ├── 🐍 sentiment_analysis.py ..... Modèle de classification
│   └── 🐍 similarity_search.py ...... Recherche vectorielle
│
├── 📂 Data/ ......................... Données du projet
│   ├── 📄 README.md ................. Documentation du dataset
│   └── 📊 Forex_data_corrected.csv .. Historique des annonces économiques
│
└── 📂 Notebooks/ .................... Expérimentation et Entraînement
    ├── 📄 README.md ................. Documentation du processus d'entraînement
    └── 📓 Construction_modele...ipynb Notebook de fine-tuning DistilRoBERTa
```

---

## 🚀 Fonctionnement Global

Le système suit un pipeline en 5 étapes pour analyser une annonce économique :

1.  **Extraction d'Embeddings** : Le texte de l'annonce est converti en vecteur par DistilRoBERTa.
2.  **Recherche de Similarité (RAG)** : Le système recherche des événements historiques similaires dans le dataset.
3.  **Analyse de Sentiment** : Le modèle classifie l'impact probable (Hausse/Baisse/Neutre).
4.  **Génération de Prompt** : Les informations (Annonce + Historique + Sentiment) sont assemblées.
5.  **Analyse LLM** : Google Gemini génère une recommandation finale en langage naturel.

---

## 🛠️ Installation et Démarrage

1.  **Prérequis** : Python 3.8+, Clé API Google Gemini.
2.  **Installation des dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
    *(Assurez-vous d'avoir un fichier requirements.txt ou installez manuellement : streamlit, pandas, torch, transformers, google-generativeai, scikit-learn)*

3.  **Lancement de l'application** :
    ```bash
    streamlit run app_interface_code/MacroLLM_app.py
    ```
