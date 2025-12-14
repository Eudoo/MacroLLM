# 💾 Données du Projet

Ce dossier contient les datasets utilisés pour l'entraînement du modèle et le fonctionnement de l'application RAG.

## 📂 Contenu

- **[Forex_data_corrected.csv](Forex_data_corrected.csv)** : Le fichier CSV principal contenant l'historique des annonces économiques et des variations de prix.

---

## 📊 Détails du Dataset

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
