import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from main_logic import clean_text, analyze_intent_score, get_sentiment
from youtube_scraper_selenium import get_youtube_comments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configuration de la page
st.set_page_config(page_title="YouTube Travel Insight Pro", layout="wide", page_icon="✈️")

# Custom CSS pour le style
st.markdown("""
    <style>
    /* Fond de la page */
    .main { background-color: #f8f9fa; }
    
    /* Correction des Metrics (Cartes) */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        border: 1px solid #eeeeee !important;
    }
    
    /* Forcer la couleur du texte à l'intérieur des metrics pour qu'il soit lisible */
    [data-testid="stMetric"] label, 
    [data-testid="stMetric"] div {
        color: #1f1f1f !important;
    }

    /* Style du bandeau pays */
    .country-header { 
        background: linear-gradient(90deg, #e0f2f1 0%, #ffffff 100%);
        padding: 25px; border-radius: 15px; margin-bottom: 25px; border-left: 8px solid #009688;
        color: #1f1f1f;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("ANALYSE PRÉDICTIVE : SENTIMENTS & INTENTIONS DE VOYAGE")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ CONFIGURATION")
    video_url = st.text_input("Lien de la vidéo YouTube", placeholder="https://www.youtube.com/watch?v=...")
    nb_scrolls = st.slider("Profondeur du scan (scrolls)", 5, 100, 20)
    st.info("Plus le nombre de scrolls est élevé, plus l'analyse sera précise mais lente.")
    analyze_btn = st.button("LANCER L'ANALYSE", use_container_width=True)

# --- LOGIQUE PRINCIPALE ---
if analyze_btn and video_url:
    with st.spinner("Extraction et analyse des commentaires en cours..."):
        raw_comments = get_youtube_comments(video_url, max_scrolls=nb_scrolls)
        
    if raw_comments:
        df = pd.DataFrame(raw_comments, columns=["comment"])
        
        # 1. Traitement de base
        df['cleaned'] = df['comment'].apply(clean_text)
        df['intent'] = df['cleaned'].apply(analyze_intent_score)
        df['sentiment'] = df['comment'].apply(get_sentiment)
        
        # 2. Entraînement du Modèle pour la prédiction avancée
        X = df['cleaned']
        y = df['intent']
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        X_vect = vectorizer.fit_transform(X)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_vect, y)
        
        # Calcul des probabilités de visite pour TOUS les commentaires
        probs = model.predict_proba(X_vect)[:, 1]
        df['prediction_score'] = probs

        # --- AFFICHAGE DES KPIs ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Commentaires", len(df))
        with col2:
            intent_count = df['intent'].sum()
            st.metric("Intentions Détectées", intent_count)
        with col3:
            avg_prob = df['prediction_score'].mean()
            st.metric("Score de Désirabilité", f"{avg_prob*100:.1f}%")
        with col4:
            sentiment_label = "Positif" if df['sentiment'].mean() > 0 else "Neutre/Négatif"
            st.metric("Ambiance Générale", sentiment_label)

        st.markdown("---")

        # --- SECTION GRAPHIQUES ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Analyse des Intentions")
            # Graphique de répartition Pie Chart
            fig_pie = px.pie(df, names='intent', 
                            title="Proportion d'intentions de visite réelles",
                            color='intent',
                            color_discrete_map={1: '#00CC96', 0: '#EF553B'},
                            hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("Prédiction du Potentiel de Voyage")
            # NOUVEAU GRAPHIQUE : Histogramme des probabilités
            fig_prob = px.histogram(df, x="prediction_score", 
                                   title="Distribution de la probabilité de visite",
                                   labels={'prediction_score': 'Probabilité de conversion (0 à 1)'},
                                   color_discrete_sequence=['#636EFA'],
                                   nbins=10)
            fig_prob.update_layout(bargap=0.1)
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        
        col_bot1, col_bot2 = st.columns(2)
        
        with col_bot1:
            st.subheader("Top Mots-Clés de Visite")
            # Extraction des mots qui influencent le plus le modèle
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            top_indices = np.argsort(coefficients)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_coeffs = [coefficients[i] for i in top_indices]
            
            fig_words = px.bar(x=top_coeffs, y=top_features, orientation='h',
                              title="Mots déclencheurs d'intention",
                              labels={'x': 'Force de prédiction', 'y': 'Mots'},
                              color=top_coeffs, color_continuous_scale='Viridis')
            st.plotly_chart(fig_words, use_container_width=True)

        with col_bot2:
            st.subheader("Testeur de Prédiction")
            user_input = st.text_input("Tapez un commentaire pour tester l'IA :", "I will book my flight tomorrow!")
            if user_input:
                vec = vectorizer.transform([clean_text(user_input)])
                score = model.predict_proba(vec)[0][1]
                
                # Jauge de prédiction
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score * 100,
                    title = {'text': "Probabilité de visite (%)"},
                    gauge = {'axis': {'range': [0, 100]},
                            'bar': {'color': "#00CC96"},
                            'steps' : [
                                {'range': [0, 50], 'color': "#ffcccb"},
                                {'range': [50, 100], 'color': "#e5f5e0"}]}))
                st.plotly_chart(fig_gauge, use_container_width=True)

        # --- DATA TABLE ---
        with st.expander("Explorer les données détaillées et prédictions"):
            # On trie par les plus susceptibles de visiter
            df_display = df.sort_values(by='prediction_score', ascending=False)
            st.dataframe(df_display[['comment', 'sentiment', 'prediction_score', 'intent']], use_container_width=True)
            st.download_button("Télécharger les prédictions (CSV)", df_display.to_csv(index=False), "predictions_voyage.csv")

    else:
        st.error("Impossible de récupérer les commentaires. La vidéo est peut-être privée ou les commentaires sont désactivés.")

else:
    # État d'attente
    st.info("**Conseil** : Utilisez des vidéos de voyage, de vlogs ou de guides touristiques pour des résultats optimaux.")
    st.image("https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1200&q=80", caption="Analyse de données touristiques")