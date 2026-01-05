import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
import feature_extraction as fe
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. DYNAMIC DATA & PERFORMANCE EVALUATION ---
@st.cache_data
def get_dataset_stats():
    """Reads actual row counts from the cleaned datasets."""
    try:
        legit_df = pd.read_csv("Dataset/structured_legitimate_list.csv")
        phish_df = pd.read_csv("Dataset/structured_phishing_list.csv")
        return len(legit_df), len(phish_df), legit_df, phish_df
    except Exception:
        return 0, 0, None, None

legit_n, phish_n, legit_df, phish_df = get_dataset_stats()
total_n = legit_n + phish_n

@st.cache_resource
def load_models():
    names = ["cnn", "lstm", "gru", "cnn_lstm", "cnn_bilstm"]
    loaded = {}
    for n in names:
        path = f"Models/{n}_model.keras"
        if os.path.exists(path):
            loaded[n] = tf.keras.models.load_model(path)
    return loaded

models = load_models()

@st.cache_data
def evaluate_models(_models, _legit_df, _phish_df):
    """Evaluates the saved models against a test sample to get real metrics."""
    if not _models or _legit_df is None:
        return pd.DataFrame()
    
    df = pd.concat([_legit_df, _phish_df], ignore_index=True).sample(frac=0.2, random_state=42)
    X = df.drop(['URL', 'label', 'dir_count', 'url_len'], axis=1).values
    y_true = df['label'].values
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    results = []
    for name, model in _models.items():
        y_prob = model.predict(X, verbose=0)
        y_pred = (y_prob > 0.5).astype(int)
        
        results.append({
            "Model": name.upper(),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred)
        })
    return pd.DataFrame(results)

# --- 2. HEADER & OBJECTIVE ---
st.title("🛡️ Detecting Phishing Websites Using Hybrid Deep Learning")

st.write(f"""
This DL-based app is developed for research purposes. Objective of the app is detecting phishing websites only using **URL Lexical and HTML Content** data. You can see the details of approach, data set, and feature set if you click on **"PROJECT DETAILS"**.
""")

# --- 3. PROJECT DETAILS (EXPANDER) ---
with st.expander("📂 PROJECT DETAILS", expanded=False):
    st.write("### Dataset Overview")
    
    if total_n > 0:
        fig_pie = px.pie(
            values=[legit_n, phish_n], 
            names=[f'Legitimate ({legit_n})', f'Phishing ({phish_n})'], 
            color_discrete_sequence=['#2ecc71', '#e74c3c'], 
            hole=0.4
        )
        st.plotly_chart(fig_pie)

        st.write("**Dataset Details (Post-Cleaning):**")
        st.write(f"- **Total Samples:** {total_n:,}")
        st.write(f"- **Legitimate Data:** {legit_n:,} URLs sourced from the Tranco Top 1M list.")
        st.write(f"- **Phishing Data:** {phish_n:,} URLs sourced from PhishTank verified archives.")
        st.write("- **Data Cleaning:** Removed duplicates, normalized URLs, and discarded non-responsive links.")

        st.divider()

        # --- FEATURE EXTRACTION SECTION ---
        st.write("### Feature Extraction")
        st.write("""
        A hybrid feature extraction approach was utilized to capture both the identity and the behavioral structure of the websites. 
        After initial extraction of 61 features, 2 biased features (`dir_count` and `url_len`) were removed to prevent data leakage, resulting in a **59-feature vector** categorized as follows:
        """)

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.write("**1. HTML Content Features**")
            st.write("- Tag counts (Inputs, Images, Buttons)")
            st.write("- Structural flags (Has Password, Has Form)")
        with f_col2:
            st.write("**2. URL Lexical Features**")
            st.write("- Domain stats (Host Length, Dot Count)")
            st.write("- Security flags (Is IP, Is Shortened)")
        with f_col3:
            st.write("**3. Behavioral Ratios**")
            st.write("- External Resource Ratios (JS, CSS, Img)")
            st.write("- Suspicious Form Action checks")

        st.divider()

        st.write("### Model Performance Comparison")
        df_results = evaluate_models(models, legit_df, phish_df)
        
        if not df_results.empty:
            st.table(df_results.set_index("Model"))
            
            fig_metrics = go.Figure()
            for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                fig_metrics.add_trace(go.Bar(x=df_results["Model"], y=df_results[metric], name=metric))
            
            fig_metrics.update_layout(barmode='group', height=500, yaxis_title="Score (0.0 - 1.0)")
            st.plotly_chart(fig_metrics, use_container_width=True)

# --- 4. LIVE DETECTION LAB ---
st.divider()
st.header("🔍 Live Detection Lab")
st.write("Submit a URL below to run real-time inference across all trained research models.")

# --- EXAMPLE URLS TABLE ---
with st.expander("💡 View Example URLs for Testing"):
    st.write("Copy and paste these URLs to see how the models respond:")
    example_data = {
        "Type": ["Legitimate", "Legitimate", "Phishing", "Phishing"],
        "URL": [
            "https://www.google.com", 
            "https://www.wikipedia.org", 
            "http://4569-5690.free.nf/verif.html", 
            "https://fr-mercdeu.firebaseapp.com/"
        ]
    }
    st.table(pd.DataFrame(example_data))

url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com")

# Define column names for feature alignment
COLUMN_NAMES = [
    'has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link',
    'has_password', 'has_email_input', 'has_hidden_element', 'has_audio', 'has_video',
    'number_of_inputs', 'number_of_buttons', 'number_of_images', 'number_of_option',
    'number_of_list', 'number_of_th', 'number_of_tr', 'number_of_href', 'number_of_paragraph',
    'number_of_script', 'length_of_title', 'has_h1', 'has_h2', 'has_h3', 'length_of_text',
    'number_of_clickable_button', 'number_of_a', 'number_of_img', 'number_of_div',
    'number_of_figure', 'has_footer', 'has_form', 'has_text_area', 'has_iframe',
    'has_text_input', 'number_of_meta', 'has_nav', 'has_object', 'has_picture',
    'number_of_sources', 'number_of_span', 'number_of_table',
    'form_action_suspicious', 'null_hyperlinks_ratio', 'external_img_ratio', 
    'external_css_ratio', 'external_js_ratio', 'url_len', 'host_len', 'dot_count', 
    'hyphen_count', 'is_ip', 'has_at', 'double_slash', 'dir_count', 'http_in_host', 
    'has_keyword', 'digit_count', 'is_shortened', 'risky_tld'
]

if st.button("Analyze URL"):
    if url_input and models:
        try:
            with st.spinner("Analyzing site structure..."):
                headers = {'User-Agent': 'Mozilla/5.0'}
                res = requests.get(url_input, timeout=5, verify=False, headers=headers)
                soup = BeautifulSoup(res.content, "html.parser")
                full_vector = fe.create_vector(soup, url_input)
                
                temp_df = pd.DataFrame([full_vector], columns=COLUMN_NAMES)
                final_df = temp_df.drop(['dir_count', 'url_len'], axis=1)
                input_data = final_df.values.reshape(1, 59, 1)

                st.write("### Model Consensus Results")
                cols = st.columns(3)
                
                for idx, (m_name, model) in enumerate(models.items()):
                    prob = model.predict(input_data, verbose=0)[0][0]
                    with cols[idx % 3]:
                        st.write(f"**{m_name.upper()}**")
                        if prob > 0.8:
                            st.error(f"🚨 Phishing ({prob*100:.1f}%)")
                        elif prob > 0.1:
                            st.warning(f"⚠️ Cautious ({prob*100:.1f}%)")
                        else:
                            st.success(f"✅ Safe ({(1-prob)*100:.1f}%)")
        except Exception as e:
            st.error(f"Analysis Failed: Ensure the URL is active and reachable. Error: {e}")