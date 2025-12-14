# 📓 Documentation Technique des Notebooks

Ce document détaille l'approche technique, l'architecture du modèle et les résultats obtenus dans le notebook **[Construction_modele_v2_DistilRoBERTa.ipynb](Construction_modele_v2_DistilRoBERTa.ipynb)**.

---

## 📋 Vue d'ensemble du projet

**MacroLLM** est un assistant d'analyse Forex basé sur l'IA qui combine un modèle **DistilRoBERTa fine-tuné** et une architecture **RAG (Retrieval-Augmented Generation)** pour interpréter les événements macroéconomiques et générer des insights de trading via **Google Gemini**.

### Objectifs
- Prédire l'impact des annonces économiques sur les paires de devises
- Fournir des analyses contextuelles basées sur l'historique du marché
- Générer des recommandations de trading actionnables
- Combiner deep learning et LLM pour une analyse hybride

---

## 🏗️ Architecture du système

### Pipeline complet

```
[Annonce économique] 
    ↓
[1. Extraction Embeddings DistilRoBERTa]
    ↓
[2. Recherche de similarité (RAG)]
    ↓
[3. Classification de sentiment (3 classes)]
    ↓
[4. Génération de prompt]
    ↓
[5. Analyse finale par Gemini LLM]
    ↓
[Recommandation de trading]
```

### Composants techniques

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Modèle de base** | DistilRoBERTa | Encodage sémantique + Classification |
| **Fine-tuning** | Hugging Face Trainer | Adaptation au domaine Forex |
| **RAG** | Cosine Similarity | Recherche de précédents historiques |
| **LLM** | Google Gemini 2.5 Flash Lite | Génération d'analyses |
| **Dataset** | 15,490 événements Forex (2007-2024) | Données d'entraînement |

---

## 📊 Dataset et préparation des données

### Source des données
- **Format** : CSV avec 12 colonnes
- **Période** : Janvier 2007 - Décembre 2024
- **Événements** : 15,490 annonces économiques
- **Devises** : USD, EUR, GBP, CAD, AUD, NZD, CHF, JPY

### Structure du dataset

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `DateTime` | Date/heure de l'annonce | 2024-06-01 19:30:00+00:00 |
| `Currency` | Devise concernée | USD, EUR, GBP... |
| `Impact` | Niveau d'impact attendu | High/Medium/Low |
| `Event` | Type d'événement | Unemployment Rate, CPI m/m |
| `Actual` | Valeur réelle | 6.9% |
| `Forecast` | Valeur prévue | 7.1% |
| `Previous` | Valeur précédente | 7.0% |
| `Price_Variation` | Variation de prix (±5min) | 0.0015 (+0.15%) |
| `Label` | Classe cible (0/1/2) | 2 (HAUSSE) |
| `Embedding_Context` | Texte formaté pour le modèle | "CAD Unemployment Rate. Actual: 6.9%, Forecast: 7.1%..." |

### Labellisation (classification ternaire)

Les labels sont créés à partir de la variation de prix observée dans les 5 minutes suivant l'annonce :

- **Label 0 (BAISSE)** : `Price_Variation < -0.00005` (variation < -0.005%)
- **Label 1 (NEUTRE)** : `-0.00005 ≤ Price_Variation ≤ +0.00005`
- **Label 2 (HAUSSE)** : `Price_Variation > +0.00005` (variation > +0.005%)

**Distribution des classes** :
- BAISSE : 31.1% (4,822 exemples)
- NEUTRE : 36.2% (5,601 exemples)
- HAUSSE : 32.7% (5,067 exemples)

### Format `Embedding_Context`

Le texte d'entrée du modèle suit ce template :
```
{Currency} {Event}. Actual: {Actual}, Forecast: {Forecast}. Usual Effect: {Usual_Effect}.
```

**Exemple** :
```
USD Non-Farm Employment Change. Actual: 250K, Forecast: 180K. 
Usual Effect: Actual greater than Forecast is good for currency.
```

---

## 🧠 Modèle DistilRoBERTa - Fine-tuning

### Choix du modèle de base

**DistilRoBERTa-base** a été choisi pour :
- **Performance** : 82M paramètres, bon compromis vitesse/qualité
- **Domaine** : Pré-entraîné sur du texte général (bien adapté aux news économiques)
- **Taille** : Plus léger que RoBERTa-large (125M param)

### Configuration du fine-tuning

```python
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,  # Mixed precision
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

### Split des données (stratifié)

- **Train** : 70% (10,843 exemples)
- **Validation** : 15% (2,323 exemples)
- **Test** : 15% (2,324 exemples)

⚠️ **Split stratifié** : garantit la même distribution de classes dans chaque ensemble.

### Performances du modèle

**Résultats sur le test set** (après 5 epochs) :

| Métrique | Score |
|----------|-------|
| **Accuracy** | 38.51% |
| **Precision (macro)** | 37.13% |
| **Recall (macro)** | 37.22% |
| **F1-Score (macro)** | 33.93% |

**Matrice de confusion** :

|        | Prédit BAISSE | Prédit NEUTRE | Prédit HAUSSE |
|--------|---------------|---------------|---------------|
| **Réel BAISSE** | 66 | 398 | 260 |
| **Réel NEUTRE** | 50 | 521 | 269 |
| **Réel HAUSSE** | 76 | 376 | 308 |

### Analyse des résultats

**Observations** :
- Le modèle a une tendance à sur-prédire la classe **NEUTRE** (plus conservateur)
- Performances limitées dues à :
  - **Complexité du problème** : le Forex est influencé par de nombreux facteurs non présents dans les données
  - **Bruit du marché** : variations de prix à court terme très volatiles
  - **Dataset** : 15k exemples peut être insuffisant pour ce type de tâche

**Utilisation** :
- Le modèle est utilisé comme **signal complémentaire**, pas comme prédicteur unique
- Les probabilités des 3 classes donnent une indication du degré de certitude
- L'intégration avec RAG et Gemini compense les limites du modèle seul

---

## 🔍 Architecture RAG (Retrieval-Augmented Generation)

### Principe

Le RAG enrichit l'analyse en recherchant des **événements historiques similaires** dans la base de données :

1. **Extraction d'embeddings** : Conversion du texte en vecteur de 768 dimensions
2. **Recherche par similarité** : Calcul de la similarité cosinus avec tous les événements passés
3. **Filtrage temporel** : Exclusion des événements postérieurs à la date de référence
4. **Sélection des top-K** : Retour des 5 événements les plus similaires

### Fonction de recherche

```python
def find_similar_events_v2(query_text, data, top_k=5, query_date=None):
    """
    Trouve les K événements historiques les plus similaires.
    
    Args:
        query_text: Texte au format Embedding_Context
        data: DataFrame avec embeddings pré-calculés
        top_k: Nombre de résultats (défaut: 5)
        query_date: Date de référence pour filtrage temporel
    
    Returns:
        DataFrame avec les événements similaires et leurs scores
    """
```

### Exemple de résultats

Pour l'annonce :
```
CAD Unemployment Rate. Actual: 6.9%, Forecast: 7.1%.
```

**Top 5 événements similaires** (avec scores) :

| Date | Event | Actual | Forecast | Réaction marché | Score |
|------|-------|--------|----------|-----------------|-------|
| 2016-06-09 | CAD Unemployment Rate | 6.9% | 7.1% | HAUSSE (+0.015%) | 1.0000 |
| 2013-10-10 | CAD Unemployment Rate | 6.9% | 7.1% | BAISSE (-0.008%) | 1.0000 |
| 2015-10-08 | CAD Unemployment Rate | 7.1% | 6.9% | BAISSE (-0.008%) | 0.9996 |
| 2013-11-07 | CAD Unemployment Rate | 6.9% | 7.0% | NEUTRE (0.001%) | 0.9998 |
| 2014-04-03 | CAD Unemployment Rate | 6.9% | 7.0% | HAUSSE (+0.009%) | 0.9998 |

### Avantages du RAG

- **Contextualisation** : Fournit des cas concrets au LLM
- **Patterns historiques** : Révèle des tendances récurrentes
- **Validation croisée** : Compare la prédiction du modèle avec l'historique
- **Transparence** : Résultats explicables et traçables

---

## 🤖 Intégration avec Google Gemini

### Modèle utilisé

**Gemini 2.5 Flash Lite** :
- Version légère et rapide de Gemini
- Optimisé pour les tâches d'analyse et de synthèse
- Limite de tokens : 2000 tokens en sortie

### Génération du prompt

Le prompt combiné inclut :

1. **L'annonce économique** (format Embedding_Context)
2. **Analyse de sentiment** du modèle DistilRoBERTa (3 classes + probabilités)
3. **Précédents historiques** trouvés par RAG (top 5 avec contexte)
4. **Instructions structurées** pour l'analyse

### Template du prompt

```
Tu es un analyste macro-économique expert spécialisé dans le marché des devises (Forex).

=== NOUVELLE ANNONCE ===
{news_text}

=== ANALYSE DE SENTIMENT (DistilRoBERTa Fine-Tuné sur Forex) ===
Prédiction : {sentiment_interpretation}
Classe prédite : {class_name}
Niveau de confiance : {confidence}
Probabilités : BAISSE={x}%, NEUTRE={y}%, HAUSSE={z}%

=== PRÉCÉDENTS HISTORIQUES SIMILAIRES ===
{historical_context}

=== TA MISSION ===
Génère une analyse complète incluant :
1. EXPLICATION : Signification de cette annonce
2. ANALYSE DE LA SURPRISE : Actual vs Forecast
3. VALIDATION DU SENTIMENT : Cohérence du modèle
4. ANALYSE HISTORIQUE : Patterns observés
5. SCÉNARIOS PROBABLES : Haussier, Baissier (avec probabilités)
6. PAIRES À SURVEILLER : Recommandations
7. RECOMMANDATION : Conseil pratique pour le trader
```

### Exemple d'analyse générée

**Input** :
```
CAD Unemployment Rate. Actual: 6.9%, Forecast: 7.1%.
```

**Output Gemini** (extrait) :
```
1. EXPLICATION
Le taux de chômage canadien est de 6.9%, inférieur aux prévisions de 7.1%. 
Cela indique une économie plus robuste que prévu.

2. ANALYSE DE LA SURPRISE
C'est une bonne surprise pour le CAD. La réduction du chômage suggère 
une économie en meilleure santé, ce qui devrait soutenir la devise.

3. VALIDATION DU SENTIMENT
Le modèle prédit HAUSSE (36.0% de confiance). Les chiffres supportent 
cette direction, bien que la confiance modérée suggère de la prudence.

4. ANALYSE HISTORIQUE
Sur 5 cas similaires, 2 ont mené à une hausse, 2 à une baisse, 1 neutre.
Les réactions passées sont mitigées, ce qui confirme la nécessité de prudence.

5. SCÉNARIOS PROBABLES
- Scénario Haussier (40-45%) : Si les autres indicateurs canadiens 
  restent positifs et que la BoC maintient une position hawkish
- Scénario Baissier (30-35%) : Si d'autres facteurs macroéconomiques 
  dominent ou si le marché a déjà intégré cette information

6. PAIRES À SURVEILLER
- USD/CAD : La plus directement impactée
- CAD/JPY : Indicateur de sentiment positif pour le CAD

7. RECOMMANDATION
Attendre une confirmation sur USD/CAD dans les 30-60 minutes. Observer 
le volume et les cassures de niveaux. Utiliser des stops loss serrés.
```

---

## 🔧 Fonction Pipeline complète

### API principale

```python
def analyze_forex_news_v2(news_text, reference_date=None, top_k=5, verbose=True):
    """
    Pipeline complet d'analyse macro-économique Forex.
    
    Args:
        news_text: Annonce au format Embedding_Context
        reference_date: Date pour filtrer les précédents (format 'YYYY-MM-DD')
        top_k: Nombre de précédents historiques à récupérer
        verbose: Afficher les détails intermédiaires
    
    Returns:
        dict: {
            'news': texte de l'annonce,
            'similar_events': DataFrame des précédents,
            'sentiment': résultat de l'analyse de sentiment,
            'prompt': prompt généré pour Gemini,
            'analysis': analyse finale générée par Gemini
        }
    """
```

### Utilisation

```python
# Exemple d'utilisation
result = analyze_forex_news_v2(
    news_text="EUR Unemployment Rate. Actual: 7.3%, Forecast: 6.6%.",
    reference_date="2024-12-01"
)

# Accès aux résultats
print(result['sentiment']['class_name'])  # NEUTRE
print(result['sentiment']['score'])       # 0.414
print(result['analysis'])                 # Analyse complète de Gemini
```

---

## 📈 Métriques et évaluation

### Performances du modèle DistilRoBERTa

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| **BAISSE** | 0.34 | 0.09 | 0.14 | 724 |
| **NEUTRE** | 0.40 | 0.62 | 0.49 | 840 |
| **HAUSSE** | 0.37 | 0.41 | 0.39 | 760 |
| **Macro avg** | 0.37 | 0.37 | 0.34 | 2324 |
| **Weighted avg** | 0.37 | 0.39 | 0.35 | 2324 |

### Interprétation

**Points forts** :
- Meilleure performance sur la classe NEUTRE (F1=0.49)
- Bonne capacité à identifier les situations sans impact fort

**Limites** :
- Difficulté à prédire les BAISSES (Recall=0.09)
- Performances globales modestes (Accuracy=38.5%)

**Explications** :
- Le Forex est un marché très complexe avec de nombreux facteurs non capturés
- Les variations à court terme (±5 min) sont très bruitées
- Le dataset, bien que conséquent, reste limité pour ce type de prédiction

### Amélioration par le système hybride

Le système complet (DistilRoBERTa + RAG + Gemini) compense les limites :
- **RAG** : Apporte du contexte historique concret
- **Gemini** : Synthétise et interprète de manière nuancée
- **Probabilités** : Fournissent une mesure d'incertitude
- **Recommandations** : Toujours accompagnées de mises en garde

---

## 💻 Environnement et dépendances

### Prérequis

- **Python** : 3.8+
- **GPU** : Recommandé (Tesla T4 utilisé dans le notebook)
- **RAM** : 16 GB minimum

### Bibliothèques principales

```
transformers==4.36.0
torch==2.1.0
datasets==2.15.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.4
tqdm==4.66.1
google-generativeai==0.3.1
```

### Installation

```bash
pip install transformers torch datasets scikit-learn pandas numpy tqdm google-generativeai
```

---

## 🚀 Guide d'utilisation

### 1. Rechargement rapide du modèle

Si vous revenez après une déconnexion et que le modèle a déjà été entraîné :

```python
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemin du modèle
MODEL_SAVE_PATH = './distilroberta_forex_final'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Modèle de classification (pour sentiment)
model_classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model_classifier = model_classifier.to(device)
model_classifier.eval()

# Modèle pour embeddings (pour similarité)
model_for_embeddings = AutoModel.from_pretrained(MODEL_SAVE_PATH)
model_for_embeddings = model_for_embeddings.to(device)
model_for_embeddings.eval()

# Dataset avec embeddings pré-calculés
data = pd.read_pickle('dataset_with_embeddings_distilroberta.pkl')
```

### 2. Analyse d'une nouvelle annonce

```python
# Définir les fonctions nécessaires (voir notebook sections 7, 8, 9, 11)

# Analyser une annonce
result = analyze_forex_news_v2(
    news_text="GBP Retail Sales m/m. Actual: -0.3%, Forecast: 0.2%. Usual Effect: Actual greater than Forecast is good for currency.",
    reference_date="2024-11-01",
    top_k=5,
    verbose=True
)
```

### 3. Accéder aux résultats

```python
# Sentiment
print(f"Prédiction : {result['sentiment']['class_name']}")
print(f"Confiance : {result['sentiment']['score']:.1%}")

# Événements similaires
print(result['similar_events'])

# Analyse finale
print(result['analysis'])
```

---

## ⚠️ Limitations et avertissements

### Limitations du système

1. **Performances du modèle** : Accuracy de 38.5%, à utiliser comme indicateur complémentaire
2. **Données historiques** : Limitées à 15k événements (2007-2024)
3. **Facteurs non capturés** : Sentiment de marché, flux d'ordres, événements géopolitiques soudains
4. **Latence** : Analyse basée sur des variations à +5 minutes (pas de trading haute fréquence)
5. **Biais du dataset** : Surreprésentation de certaines devises (USD, EUR)

### Avertissements pour le trading

⚠️ **Ce système est un outil d'aide à la décision, PAS un système de trading automatique.**

- Les prédictions ne garantissent pas les résultats futurs
- Toujours utiliser une gestion du risque appropriée (stop loss, position sizing)
- Ne jamais trader uniquement sur la base de ces prédictions
- Considérer le contexte macroéconomique global
- Les marchés peuvent réagir de manière irrationnelle à court terme

### Recommandations d'utilisation

✅ **Bonnes pratiques** :
- Utiliser comme confirmation d'une analyse technique
- Attendre 30-60 minutes après une annonce pour voir la réaction du marché
- Croiser avec d'autres sources d'information
- Tester sur compte démo avant tout trading réel

❌ **À éviter** :
- Trading immédiat sur la seule base des prédictions
- Ignorer les niveaux techniques clés (support/résistance)
- Prendre des positions sans stop loss
- Over-leveraging

---

## 🔮 Améliorations futures

### Court terme

1. **Augmentation du dataset** : Collecter plus d'événements (objectif : 50k+)
2. **Feature engineering** : Ajouter des variables techniques (volatilité, volume)
3. **Ensembling** : Combiner plusieurs modèles (BERT, RoBERTa, FinBERT)
4. **Fine-tuning Gemini** : Personnaliser les prompts pour des analyses plus ciblées

### Moyen terme

1. **Modèle multi-horizon** : Prédire à 5min, 15min, 1h, 4h
2. **Intégration sentiment Twitter/Reddit** : Analyser le sentiment social
3. **API temps réel** : Connexion à des flux de données live
4. **Interface web** : Dashboard interactif avec visualisations

### Long terme

1. **Reinforcement Learning** : Agent apprenant par trading simulé
2. **Explainability (XAI)** : Interprétation des décisions du modèle (SHAP, LIME)
3. **Multi-asset** : Extension aux actions, crypto, commodities
4. **Backtesting rigoureux** : Validation sur 10+ années de données

---

## 📚 Références et ressources

### Papers et articles

- **RoBERTa** : Liu et al., 2019 - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- **DistilBERT** : Sanh et al., 2019 - "DistilBERT, a distilled version of BERT"
- **RAG** : Lewis et al., 2020 - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

### Datasets Forex

- **Forex Factory Calendar** : https://www.forexfactory.com/calendar
- **Investing.com Economic Calendar** : https://www.investing.com/economic-calendar

### Outils et bibliothèques

- **Hugging Face Transformers** : https://huggingface.co/docs/transformers
- **Google Gemini API** : https://ai.google.dev/
- **Scikit-learn** : https://scikit-learn.org/

---

## 👥 Crédits

**Auteurs** : Eudes CODO, Emmanuella GBODO, Grâce WILSON
**Projet** : MacroLLM - Assistant d'Analyse Forex IA  
**Date** : Décembre 2025  

---

## 📧 Contact et support

Pour toute question ou suggestion :
- **Email** : eudescodo00@gmail.com
- **GitHub** : https://github.com/Eudoo/MacroLLM.git

---

*Cette documentation a été générée pour le projet MacroLLM, un système hybride combinant DistilRoBERTa fine-tuné, RAG et Google Gemini pour l'analyse du marché Forex.*
