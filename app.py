import streamlit as st
import tensorflow as tf
import pickle
import os
import re
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI") 
if not MONGO_URI and "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]

if not MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/"

try:
    client = MongoClient(MONGO_URI)
    db = client['phishing_detector_db']
    url_collection = db['url_logs']
    list_collection = db['whitelists_blacklists']
    client.server_info()
except Exception as e:
    st.error(f"❌ Database Connection Failed: {e}")
    st.stop()

# File Paths
MODEL_PATH = 'Models/CNN-BiLSTM_best.h5'
TOKENIZER_PATH = 'Dataset/tokenizer.pkl'
DATA_PATH = 'Dataset/processed_data.csv' 
MAX_LEN = 200

# --- Helper Functions ---

@st.cache_resource
def load_resources():
    """Loads the model and tokenizer."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

@st.cache_data
def load_dataset():
    """Loads and caches the CSV dataset for display."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def normalize_url(url):
    if not url: return ""
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    if url.endswith('/'):
        url = url[:-1]
    return url

def predict_url(model, tokenizer, url):
    norm_url = normalize_url(url)
    seq = [tokenizer.get(c, 0) for c in norm_url]
    padded_seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')
    return model.predict(padded_seq)[0][0]

def save_log(url, status, confidence, source, reviewed=False):
    log_entry = {
        "url": url,
        "normalized_url": normalize_url(url),
        "status": status,
        "confidence": float(confidence),
        "source": source,
        "timestamp": datetime.utcnow(),
        "reviewed": reviewed
    }
    url_collection.insert_one(log_entry)

# --- UI Layout ---

st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Admin Login"])

model, tokenizer = load_resources()

# --- PAGE 1: Homepage ---
if page == "Homepage":
    st.title("🛡️ Phishing Website Detector")
    st.markdown("Enter a website URL below to check if it's safe using our **Deep Learning (CNN-BiLSTM)** model.")

    url_input = st.text_input("Website URL", placeholder="http://example.com")

    if st.button("Scan Now"):
        if not url_input:
            st.warning("Please enter a URL.")
        else:
            norm_url = normalize_url(url_input)
            
            # Check MongoDB Whitelist/Blacklist
            list_entry = list_collection.find_one({"url": norm_url})
            
            if list_entry:
                if list_entry['type'] == 'whitelist':
                    st.success(f"✅ **Legitimate** (Verified Whitelist)")
                    save_log(url_input, "Legitimate", 1.0, "Whitelist", True)
                else:
                    st.error(f"🚨 **Phishing** (Verified Blacklist)")
                    save_log(url_input, "Phishing", 1.0, "Blacklist", True)
            
            else:
                # Use AI Model
                if model and tokenizer:
                    with st.spinner("Analyzing patterns..."):
                        prob = predict_url(model, tokenizer, url_input)
                    
                    if prob >= 0.90:
                        st.error(f"🚨 **Phishing Detected**")
                        st.metric("Confidence Score", f"{prob*100:.2f}%")
                        st.info("This site exhibits patterns commonly found in phishing attacks.")
                        save_log(url_input, "Phishing", prob, "Model", True)
                        
                    elif prob <= 0.10:
                        st.success(f"✅ **Legitimate Site**")
                        st.metric("Confidence Score", f"{(1-prob)*100:.2f}%")
                        st.info("This site looks safe based on our analysis.")
                        save_log(url_input, "Legitimate", prob, "Model", True)
                        
                    else:
                        st.warning(f"⚠️ **Uncertain / Suspicious**")
                        st.metric("Phishing Probability", f"{prob*100:.2f}%")
                        st.write("Our model is not 100% sure. This URL has been flagged for manual review.")
                        save_log(url_input, "Pending Review", prob, "Model", False)
                else:
                    st.error("Model failed to load.")
    
    # Data Display Section
    st.divider()
    st.subheader("Known Sites Database")
    st.markdown("A list of known legitimate and phishing websites from our training dataset.")
    
    df = load_dataset()
    if df is not None:
        col1, col2 = st.columns(2)
        
        # Filter and Fix Index
        # Reset index to default (0,1,2...), then add 1 to make it start at 1
        legit_sites = df[df['label'] == 0][['url']].reset_index(drop=True)
        legit_sites.index += 1
        
        phishing_sites = df[df['label'] == 1][['url']].reset_index(drop=True)
        phishing_sites.index += 1
        
        with col1:
            # Removed the (count) from header
            st.success("**Legitimate Sites**")
            st.dataframe(legit_sites, use_container_width=True, height=400)
            
        with col2:
            # Removed the (count) from header
            st.error("**Phishing Sites**")
            st.dataframe(phishing_sites, use_container_width=True, height=400)
    else:
        st.warning("Dataset not found. Please ensure 'processed_data.csv' is in the Dataset folder.")

# --- PAGE 2: Admin Login ---
elif page == "Admin Login":
    st.title("Admin Sign In")
    st.write("Enter your credentials to access your account")
    
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            admin_user = os.getenv("ADMIN_USER") or (st.secrets["ADMIN_USER"] if "ADMIN_USER" in st.secrets else "admin")
            admin_pass = os.getenv("ADMIN_PASS") or (st.secrets["ADMIN_PASS"] if "ADMIN_PASS" in st.secrets else "123")
            
            if username == admin_user and password == admin_pass:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success("Logged in as Admin")
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.divider()
        st.subheader("Admin Dashboard")
        
        # --- Pending Reviews ---
        st.write("#### ⚠️ Pending Reviews")
        pending_items = list(url_collection.find({"reviewed": False}).sort("timestamp", -1))
        
        if not pending_items:
            st.info("No URLs pending review.")
        
        for item in pending_items:
            with st.expander(f"{item['url']} (Score: {item['confidence']:.2f})"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Mark Legitimate", key=f"legit_{item['_id']}"):
                        url_collection.update_one({"_id": item['_id']}, {"$set": {"status": "Legitimate", "reviewed": True}})
                        list_collection.update_one(
                            {"url": item['normalized_url']}, 
                            {"$set": {"url": item['normalized_url'], "type": "whitelist"}}, 
                            upsert=True
                        )
                        st.success("Whitelisted!")
                        st.rerun()
                        
                with col2:
                    if st.button("Mark Phishing", key=f"phish_{item['_id']}"):
                        url_collection.update_one({"_id": item['_id']}, {"$set": {"status": "Phishing", "reviewed": True}})
                        list_collection.update_one(
                            {"url": item['normalized_url']}, 
                            {"$set": {"url": item['normalized_url'], "type": "blacklist"}}, 
                            upsert=True
                        )
                        st.error("Blacklisted!")
                        st.rerun()

        st.divider()
        
        # --- History ---
        st.write("#### 📋 Recent Scan History")
        recent_logs = list(url_collection.find().sort("timestamp", -1).limit(20))
        
        if recent_logs:
            df_logs = pd.DataFrame(recent_logs)
            st.dataframe(df_logs[['timestamp', 'url', 'status', 'confidence', 'source']], use_container_width=True)
        else:
            st.write("No history found.")