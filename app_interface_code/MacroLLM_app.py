import streamlit as st
import time
import pandas as pd
import sys
import os

# --- Configuration de la page (Mode Large) ---
# DOIT ETRE LA PREMIERE COMMANDE STREAMLIT
st.set_page_config(
    page_title="MacroLLM - Forex Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le dossier courant au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pipeline_RAG import analyze_forex_news_v2
except ImportError as e:
    st.error(f"Erreur critique : Impossible d'importer 'pipeline_RAG'.\n\nDétails : {e}")
    st.stop()

# --- CSS Personnalisé (Look Dashboard Financier) ---
st.markdown("""
<style>
    /* Fond général */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* En-tête */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3a8a; /* Bleu foncé professionnel */
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    /* Cartes de résultats */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    
    /* Style des onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Bouton d'analyse */
    .stButton button {
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar : Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("🧠 Paramètres RAG")
    top_k = st.slider("Nombre de précédents (Top-K)", 1, 10, 5)
    
    st.markdown("---")
    st.info("ℹ️ **Statut :** Pipeline RAG Connecté ✅")

# --- En-tête Principal ---
st.markdown("<div class='main-header'><h1>🌍 MacroLLM : Assistant Forex IA <img src='https://img.icons8.com/fluency/96/bullish.png' width='50' style='vertical-align: middle;'/></h1><p>Analyse d'impact macro-économique par Intelligence Artificielle & RAG</p></div>", unsafe_allow_html=True)

# --- Zone de Saisie ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📰 Annonce Économique")
    news_input = st.text_area(
        "Collez l'annonce ici (Format Forex Factory) :",
        placeholder="Ex: USD CPI m/m. Actual: 0.4%, Forecast: 0.2%. Usual Effect: Actual greater than Forecast is good for currency...",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 📅 Contexte")
    date_input = st.date_input("Date de l'annonce", value=pd.Timestamp.now())
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 LANCER L'ANALYSE", use_container_width=True)

# --- Zone de Résultats ---
if analyze_btn and news_input:
    
    # 1. Animation de chargement et Exécution du Pipeline
    with st.status("🧠 Analyse en cours...", expanded=True) as status:
        st.write("🚀 Initialisation du pipeline RAG...")
        start_time = time.time()
        
        try:
            # Appel réel au pipeline
            st.write("🔍 Recherche vectorielle et Analyse de sentiment...")
            result = analyze_forex_news_v2(
                news_text=news_input,
                reference_date=date_input,
                top_k=top_k,
                verbose=False
            )
            
            st.write("🤖 Génération de l'analyse avec Gemini...")
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            status.update(label="✅ Analyse Terminée !", state="complete", expanded=False)
            
            # Extraction des résultats
            sentiment = result['sentiment']
            analysis_text = result['analysis']
            similar_df = result['similar_events']

            st.divider()

            # 2. Métriques Clés (Synthèse)
            st.subheader("📊 Synthèse de l'Impact")
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(
                    label="Sentiment IA", 
                    value=f"{sentiment['class_name']}", 
                    delta=f"Confiance: {sentiment['score']:.1%}"
                )
            with m2:
                # Affichage de la probabilité la plus forte
                max_prob = max(sentiment['all_probabilities'].values())
                st.metric(
                    label="Probabilité Dominante", 
                    value=f"{max_prob:.1%}", 
                    delta="Modèle DistilRoBERTa"
                )
            with m3:
                st.metric(label="Précédents Trouvés", value=f"{len(similar_df)} événements")
            with m4:
                st.metric(label="Temps d'inférence", value=f"{elapsed_time:.2f}s")

            # 3. Onglets Détaillés
            tab_analysis, tab_history = st.tabs(["🤖 Analyse Stratégique", "📜 Historique Similaire"])

            with tab_analysis:
                # Affichage Markdown direct de la réponse de Gemini
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">
                    {analysis_text}
                </div>
                """, unsafe_allow_html=True)

            with tab_history:
                st.write("Voici les événements passés les plus similaires trouvés dans la base vectorielle :")
                
                # Formatage du dataframe pour l'affichage
                if not similar_df.empty:
                    display_df = similar_df.copy()
                    # Sélection et renommage des colonnes pour l'affichage
                    cols_to_show = ['DateTime', 'Event', 'Actual', 'Forecast', 'Price_Variation', 'Similarity_Score']
                    # Vérifier si les colonnes existent (au cas où)
                    cols_existing = [c for c in cols_to_show if c in display_df.columns]
                    display_df = display_df[cols_existing]
                    
                    st.dataframe(
                        display_df.style.format({
                            'Price_Variation': '{:.4f}',
                            'Similarity_Score': '{:.4f}'
                        }), 
                        use_container_width=True
                    )
                else:
                    st.info("Aucun événement similaire trouvé.")

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'analyse : {str(e)}")
            status.update(label="❌ Erreur", state="error")

elif analyze_btn and not news_input:
    st.warning("⚠️ Veuillez coller une annonce économique pour démarrer l'analyse.")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.8em;'>MacroLLM v2.0 • Développé pour le cours de NLP 3e Année</div>", unsafe_allow_html=True)