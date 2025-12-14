import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
import os
import sys

# Import de la configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import HF_MODEL_ID

warnings.filterwarnings('ignore')

# Configuration du chemin du modèle (Hugging Face Hub ou local)
MODEL_PATH = HF_MODEL_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⏳ Chargement du modèle d'embeddings depuis {MODEL_PATH}...")

try:
    # Chargement du tokenizer et du modèle
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_for_embeddings = AutoModel.from_pretrained(MODEL_PATH)
    model_for_embeddings = model_for_embeddings.to(device)
    model_for_embeddings.eval()
    print("✅ Modèle d'embeddings chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle d'embeddings : {e}")
    print(f"Assurez-vous que le modèle '{MODEL_PATH}' est accessible (Hugging Face ou local).")

def get_embeddings_distilroberta(texts, batch_size=32):
    """
    Extrait les embeddings avec notre DistilRoBERTa fine-tuné.
    Utilise Mean Pooling sur les tokens.
    """
    if 'model_for_embeddings' not in globals() or 'tokenizer' not in globals():
        return []

    all_embeddings = []

    # Si texts est une chaine unique, la mettre dans une liste
    if isinstance(texts, str):
        texts = [texts]

    # Utiliser tqdm seulement si on a beaucoup de textes
    iterator = range(0, len(texts), batch_size)
    if len(texts) > batch_size:
        iterator = tqdm(iterator, desc="Extraction embeddings")

    for i in iterator:
        batch_texts = texts[i:i + batch_size]

        # Tokenisation
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        encoded = {key: val.to(device) for key, val in encoded.items()}

        # Inférence
        with torch.no_grad():
            outputs = model_for_embeddings(**encoded)

        # Mean Pooling
        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        all_embeddings.extend(mean_embeddings.cpu().numpy())

    return all_embeddings

def find_similar_events_v2(query_text, data, top_k=5, query_date=None):
    """
    Trouve les K événements historiques les plus similaires (Version DistilRoBERTa).

    Args:
        query_text: Texte au format Embedding_Context
        data: DataFrame avec la colonne Embedding
        top_k: Nombre de résultats
        query_date: Date de référence pour filtrage temporel
    """
    # Gestion de la date
    if query_date is None:
        reference_date = pd.Timestamp.now()
    elif isinstance(query_date, str):
        reference_date = pd.to_datetime(query_date)
    else:
        reference_date = pd.to_datetime(query_date)

    # Filtrage temporel
    data_filtered = data.copy()
    # Assurer que DateTime est bien parsé
    if 'DateTime_parsed' not in data_filtered.columns:
        data_filtered['DateTime_parsed'] = pd.to_datetime(data_filtered['DateTime'], errors='coerce')

    # Supprimer le timezone si présent pour éviter l'erreur de comparaison
    if data_filtered['DateTime_parsed'].dt.tz is not None:
        data_filtered['DateTime_parsed'] = data_filtered['DateTime_parsed'].dt.tz_localize(None)
    if hasattr(reference_date, 'tz') and reference_date.tz is not None:
        reference_date = reference_date.tz_localize(None)

    data_filtered = data_filtered[data_filtered['DateTime_parsed'] < reference_date]

    if len(data_filtered) == 0:
        print(f"⚠️ Aucun événement trouvé avant {reference_date}")
        return pd.DataFrame()

    # Vectorisation avec notre modèle
    query_embedding = get_embeddings_distilroberta([query_text], batch_size=1)[0]

    # Similarité cosinus
    embeddings_matrix = np.vstack(data_filtered['Embedding'].values)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = data_filtered.iloc[top_indices].copy()
    results['Similarity_Score'] = similarities[top_indices]

    # Colonnes à retourner
    cols_to_return = ['DateTime', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous', 'Price_Variation', 'Label', 'Similarity_Score']
    # Vérifier si les colonnes existent
    existing_cols = [c for c in cols_to_return if c in results.columns]
    
    return results[existing_cols]
