# Configuration du projet MacroLLM

# -----------------------------------------------------------------------------
# CONFIGURATION HUGGING FACE
# -----------------------------------------------------------------------------
# Remplacez par votre identifiant Hugging Face et le nom de votre modèle
HF_MODEL_ID = "MacroLLM/MacroLLM-DistilRoBERTa-Forex"

# Configuration du Dataset sur Hugging Face (pour le fichier pickle)
# Par défaut, on utilise le même dépôt que le modèle
HF_DATASET_REPO_ID = HF_MODEL_ID
HF_DATASET_FILENAME = "dataset_with_embeddings_distilroberta.pkl"

# Si vous testez en local avec le dossier, vous pouvez laisser le chemin relatif
# HF_MODEL_ID = "../distilroberta_forex_final" 

# -----------------------------------------------------------------------------
# CONFIGURATION GEMINI
# -----------------------------------------------------------------------------
import streamlit as st
import os

# Gestion sécurisée de la clé API pour le déploiement
# 1. Essaie de charger depuis les secrets Streamlit (Cloud)
# 2. Sinon, utilise la variable d'environnement
# 3. Sinon, utilise la clé en dur (Déconseillé en production publique)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBExigWwXDNDc_J9X2wtQwvLimfIfX_jYA")

# -----------------------------------------------------------------------------
# CHEMINS DES DONNÉES
# -----------------------------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Forex_data_corrected.csv')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'dataset_with_embeddings_distilroberta.pkl')
