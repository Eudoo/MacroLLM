import pandas as pd
import google.generativeai as genai
import os
import sys
from huggingface_hub import hf_hub_download

# Ajouter le dossier courant au path pour les imports si nécessaire
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analysis import get_sentiment_distilroberta
from similarity_search import find_similar_events_v2
from config import GEMINI_API_KEY, HF_DATASET_REPO_ID, HF_DATASET_FILENAME

# Configuration Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialisation du modèle Gemini
model_gemini = genai.GenerativeModel(
    'gemini-2.5-flash-lite',
    system_instruction="Tu dois TOUJOURS commencer ta réponse par cette phrase exacte : 'En tant qu'analyste macro-économique spécialisé dans le Forex, voici mon analyse et mes recommandations concernant cette annonce :'"
)

# Chargement des données (Dataset avec embeddings)
print(f"⏳ Chargement du dataset...")

try:
    # Essayer de télécharger depuis Hugging Face
    print(f"⬇️ Tentative de téléchargement depuis Hugging Face ({HF_DATASET_REPO_ID})...")
    data_path = hf_hub_download(repo_id=HF_DATASET_REPO_ID, filename=HF_DATASET_FILENAME)
    print(f"✅ Fichier téléchargé : {data_path}")
    
    data = pd.read_pickle(data_path)
    print(f"✅ Dataset chargé : {len(data)} lignes")

except Exception as e:
    print(f"⚠️ Impossible de télécharger depuis Hugging Face : {e}")
    print("🔄 Tentative de chargement local...")
    
    # Fallback local
    LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_with_embeddings_distilroberta.pkl')
    
    if os.path.exists(LOCAL_DATA_PATH):
        try:
            data = pd.read_pickle(LOCAL_DATA_PATH)
            print(f"✅ Dataset local chargé : {len(data)} lignes")
        except Exception as local_e:
            print(f"❌ Erreur lors du chargement du dataset local : {local_e}")
            data = pd.DataFrame()
    else:
        print(f"❌ Erreur : Le fichier est introuvable localement ({LOCAL_DATA_PATH}) et sur Hugging Face.")
        data = pd.DataFrame()

def generate_analysis_v2(news_text, similar_events_df, sentiment_result):
    """
    Génère un prompt d'analyse macro-économique complète pour un LLM.
    VERSION 2 : Utilise DistilRoBERTa fine-tuné avec classification ternaire.

    Args:
        news_text: La nouvelle news au format Embedding_Context
        similar_events_df: DataFrame des événements historiques similaires
        sentiment_result: Résultat de l'analyse de sentiment DistilRoBERTa (3 classes)

    Returns:
        str: Le prompt formaté pour le LLM
    """
    # Construction du contexte historique avec les labels
    historical_context = ""
    for idx, row in similar_events_df.iterrows():
        # Direction basée sur Price_Variation
        if row['Price_Variation'] > 0.00005:
            direction = "HAUSSE"
        elif row['Price_Variation'] < -0.00005:
            direction = "BAISSE"
        else:
            direction = "NEUTRE"

        variation = abs(row['Price_Variation'] * 100)

        # Gestion du format de date
        date_value = row['DateTime']
        if hasattr(date_value, 'strftime'):
            date_str = date_value.strftime('%Y-%m-%d')
        else:
            date_str = str(date_value)[:10]

        # Label historique
        label_map = {0: "BAISSE", 1: "NEUTRE", 2: "HAUSSE"}
        label_hist = label_map.get(row.get('Label', 1), "N/A")

        historical_context += f"""
    - {date_str} : {row['Event']} ({row['Currency']})
      Actual: {row['Actual']}, Forecast: {row['Forecast']}, Previous: {row['Previous']}
      Réaction marché : {direction} ({variation:.4f}%) - Label: {label_hist}
      Score de similarité : {row['Similarity_Score']:.4f}
"""

    # Interprétation du sentiment DistilRoBERTa (3 classes)
    class_name = sentiment_result['class_name']
    confidence = sentiment_result['score']
    probs = sentiment_result['all_probabilities']

    if class_name == "HAUSSE":
        sentiment_interpretation = "HAUSSIER pour la devise (signal d'achat potentiel)"
    elif class_name == "BAISSE":
        sentiment_interpretation = "BAISSIER pour la devise (signal de vente potentiel)"
    else:
        sentiment_interpretation = "NEUTRE (pas d'impact significatif attendu)"

    # Construction du prompt pour le LLM
    prompt = f"""Tu es un analyste macro-économique expert spécialisé dans le marché des devises (Forex).
Un trader te demande d'analyser l'annonce économique suivante et de lui donner des conseils.

=== NOUVELLE ANNONCE ===
{news_text}

=== ANALYSE DE SENTIMENT (DistilRoBERTa Fine-Tuné sur Forex) ===
Prédiction : {sentiment_interpretation}
Classe prédite : {class_name}
Niveau de confiance : {confidence:.1%}

Probabilités détaillées :
- BAISSE : {probs['BAISSE']:.1%}
- NEUTRE : {probs['NEUTRE']:.1%}
- HAUSSE : {probs['HAUSSE']:.1%}

Note : Ce modèle a été fine-tuné sur des données Forex réelles avec des labels basés sur les vraies réactions du marché (Price_Variation).

=== PRÉCÉDENTS HISTORIQUES SIMILAIRES (trouvés par DistilRoBERTa Embeddings) ===
{historical_context}

=== TA MISSION ===
En te basant sur l'analyse de sentiment ET les précédents historiques, génère une analyse complète qui inclut :

1. **EXPLICATION** : Explique simplement ce que signifie cette annonce économique.

2. **ANALYSE DE LA SURPRISE** : Compare Actual vs Forecast. Est-ce une bonne ou mauvaise surprise pour la devise ?

3. **VALIDATION DU SENTIMENT** : La prédiction du modèle ({class_name}) est-elle cohérente avec les chiffres annoncés ?

4. **ANALYSE HISTORIQUE** : Que s'est-il passé dans le passé avec des annonces similaires ? Quel pattern observes-tu ?

5. **SCÉNARIOS PROBABLES** :
   - Scénario Haussier (pour la devise) : conditions et probabilité
   - Scénario Baissier (pour la devise) : conditions et probabilité

6. **PAIRES À SURVEILLER** : Quelles paires de devises surveiller en priorité ?

7. **RECOMMANDATION** : Conseil pratique pour le trader (timing, prudence, confirmation à attendre, etc.)

Réponds de manière structurée et professionnelle.
"""

    return prompt

def call_llm(prompt):
    """
    Appelle l'API Gemini pour générer une analyse.

    Args:
        prompt: Le prompt formaté avec le contexte DistilRoBERTa

    Returns:
        str: La réponse générée par Gemini
    """
    try:
        response = model_gemini.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000,
            )
        )
        return response.text
    except Exception as e:
        return f"Erreur lors de l'appel à Gemini : {str(e)}"

def analyze_forex_news_v2(news_text, reference_date=None, top_k=5, verbose=True):
    """
    🚀 PIPELINE COMPLET V2 - Assistant Macro-Économique Forex

    Combine :
    1. DistilRoBERTa Embeddings (recherche de similarité)
    2. DistilRoBERTa Classifier (sentiment 3 classes)
    3. Gemini LLM (génération d'analyse)

    Args:
        news_text: Annonce au format Embedding_Context
                   Ex: "USD CPI m/m. Actual: 0.4%, Forecast: 0.2%. Usual Effect: ..."
        reference_date: Date pour filtrer les précédents (format 'YYYY-MM-DD')
        top_k: Nombre de précédents historiques à récupérer
        verbose: Afficher les détails intermédiaires

    Returns:
        dict: Résultats complets (similar_events, sentiment, analysis)
    """
    if verbose:
        print("=" * 70)
        print("🚀 ASSISTANT MACRO-ÉCONOMIQUE FOREX V2 (DistilRoBERTa + Gemini)")
        print("=" * 70)
        print(f"\n📝 Annonce : {news_text[:100]}...")

    # ÉTAPE 1 : Recherche de précédents
    if verbose:
        print("\n📊 Étape 1 : Recherche de précédents historiques...")
    
    # Utilisation de la variable globale 'data' chargée au début du script
    if data.empty:
        print("⚠️ Attention : Dataset vide ou non chargé.")
        similar_events = pd.DataFrame()
    else:
        similar_events = find_similar_events_v2(news_text, data, top_k=top_k, query_date=reference_date)

    # ÉTAPE 2 : Analyse de sentiment
    if verbose:
        print("\n🧠 Étape 2 : Analyse de sentiment (3 classes)...")
    sentiment = get_sentiment_distilroberta(news_text)
    if verbose:
        print(f"   → Prédiction : {sentiment['class_name']} ({sentiment['score']:.1%})")

    # ÉTAPE 3 : Génération du prompt
    if verbose:
        print("\n📝 Étape 3 : Génération du prompt...")
    prompt = generate_analysis_v2(news_text, similar_events, sentiment)

    # ÉTAPE 4 : Appel à Gemini
    if verbose:
        print("\n🤖 Étape 4 : Appel à Gemini...")
    analysis = call_llm(prompt)

    if verbose:
        print("\n" + "=" * 70)
        print("📋 ANALYSE FINALE")
        print("=" * 70)
        print(analysis)

    return {
        'news': news_text,
        'similar_events': similar_events,
        'sentiment': sentiment,
        'prompt': prompt,
        'analysis': analysis
    }

if __name__ == "__main__":
    # Test simple si le script est exécuté directement
    test_query = "USD CPI m/m. Actual: 0.4%, Forecast: 0.2%. Usual Effect: Actual greater than Forecast is good for currency."
    print(f"Test avec : {test_query}")
    analyze_forex_news_v2(test_query)


