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
# Assurez-vous que cette clé est définie dans vos variables d'environnement
# ou dans les secrets de Streamlit (.streamlit/secrets.toml) pour la sécurité.
GEMINI_API_KEY = "AIzaSyBExigWwXDNDc_J9X2wtQwvLimfIfX_jYA"

# -----------------------------------------------------------------------------
# CHEMINS DES DONNÉES
# -----------------------------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Forex_data_corrected.csv')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'dataset_with_embeddings_distilroberta.pkl')
