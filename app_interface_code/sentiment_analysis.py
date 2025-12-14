import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration du chemin du modèle (relatif au dossier 'code final')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'distilroberta_forex_final')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⏳ Chargement du modèle de sentiment depuis {MODEL_PATH}...")

try:
    # Chargement du tokenizer et du modèle
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model_classifier = model_classifier.to(device)
    model_classifier.eval()
    print("✅ Modèle de sentiment chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle de sentiment : {e}")
    print("Assurez-vous que le dossier 'distilroberta_forex_final' existe à la racine du projet.")

def get_sentiment_distilroberta(text):
    """
    Prédit le sentiment (BAISSE/NEUTRE/HAUSSE) avec notre modèle fine-tuné.

    Args:
        text: Texte au format Embedding_Context
              Ex: "EUR Final Manufacturing PMI. Actual: 56.5, Forecast: 56.5. Usual Effect: ..."

    Returns:
        dict: {'label': 'negative/neutral/positive', 'score': 0.0-1.0, ...}
    """
    if 'model_classifier' not in globals() or 'tokenizer' not in globals():
        return {"error": "Modèle non chargé"}

    # Tokenisation
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    encoded = {key: val.to(device) for key, val in encoded.items()}

    # Prédiction
    with torch.no_grad():
        outputs = model_classifier(**encoded)

    # Softmax pour obtenir les probabilités
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()

    # Mapping vers le format attendu (compatible avec le pipeline RAG)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label = label_map[predicted_class]

    # Détails pour affichage
    class_names = {0: "BAISSE", 1: "NEUTRE", 2: "HAUSSE"}

    return {
        'label': label,
        'score': confidence,
        'class_name': class_names[predicted_class],
        'all_probabilities': {
            'BAISSE': probs[0][0].item(),
            'NEUTRE': probs[0][1].item(),
            'HAUSSE': probs[0][2].item()
        }
    }
