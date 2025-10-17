import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Spotify Song Popularity Predictor", layout="wide")

st.title("🎵 Spotify Song Popularity Dashboard")
st.write("This Streamlit app analyzes popular Spotify songs and builds a Random Forest model to predict **mode** (Major/Minor).")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # --- Data Preprocessing ---
    df = df.drop(columns=['in_apple_playlists', 'key', 'in_deezer_playlists', 
                          'in_shazam_charts', 'in_apple_charts', 'valence_%',
                          'acousticness_%', 'instrumentalness_%', 'liveness_%',
                          'speechiness_%', 'artist_count', 'in_spotify_playlists'], errors='ignore')

    le = LabelEncoder()
    if 'track_name' in df.columns:
        df['track_name'] = le.fit_transform(df['track_name'])
    if 'mode' in df.columns:
        df['mode'] = le.fit_transform(df['mode'])

    # --- Define Features and Target ---
    if 'mode' in df.columns:
        X = df.drop('mode', axis=1)
        y = df['mode']

        # --- Train-Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Train Model ---
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # --- Model Evaluation ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("✅ Model Evaluation")
        st.write(f"**Accuracy:** {acc:.2f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # --- Feature Importance ---
        st.subheader("🌲 Feature Importance")
        importances = model.feature_importances_
        feature_names = X.columns
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax2)
        st.pyplot(fig2)
    else:
        st.error("❌ 'mode' column not found in dataset.")
else:
    st.info("👆 Upload your `Popular_Spotify_Songs.csv` file to begin analysis.")

