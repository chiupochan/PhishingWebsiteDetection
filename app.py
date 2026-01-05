import streamlit as st
import tensorflow as tf
import pickle
import os
import re
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# MongoDB Connection
# Checks for 'MONGO_URI' in .env or Streamlit Secrets
MONGO_URI = os.getenv("MONGO_URI") 
if not MONGO_URI and "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]

# Default fallback (local) if nothing is found
if not MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/"

try:
    client = MongoClient(MONGO_URI)
    db = client['phishing_detector_db']
    url_collection = db['url_logs']
    list_collection = db['whitelists_blacklists']
    # Test connection
    client.server_info()
except Exception as e:
    st.error(f"❌ Database Connection Failed: {e}")
    st.stop()

# Model Paths
MODEL_PATH = 'models/CNN-BiLSTM.h5'
TOKENIZER_PATH = 'dataset/tokenizer.pkl'
MAX_LEN = 200

# --- Helper Functions ---

@st.cache_resource
def load_resources():
    """Loads the model and tokenizer only once to speed up the app."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

def normalize_url(url):
    if not url: return ""
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    if url.endswith('/'):
        url = url[:-1]
    return url

def predict_url(model, tokenizer, url):
    # 1. Normalize
    norm_url = normalize_url(url)
    
    # 2. Tokenize (Character Level)
    # Convert chars to integers using the dictionary
    seq = [tokenizer.get(c, 0) for c in norm_url]
    
    # 3. Pad Sequence (The Correct Way)
    # We pass [seq] because pad_sequences expects a list of sequences
    padded_seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Predict
    # Returns probability between 0 and 1
    return model.predict(padded_seq)[0][0]

def save_log(url, status, confidence, source, reviewed=False):
    """Saves the scan result to MongoDB"""
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

st.set_page_config(page_title="Phishing Detector AI", page_icon="🛡️", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Scanner", "Admin Dashboard"])

model, tokenizer = load_resources()

# --- PAGE 1: URL Scanner ---
if page == "Scanner":
    st.title("🛡️ AI Phishing URL Scanner")
    st.markdown("Enter a website URL below to check if it's safe using our **Deep Learning (CNN-BiLSTM)** model.")

    url_input = st.text_input("Website URL", placeholder="http://example.com")

    if st.button("Scan Now"):
        if not url_input:
            st.warning("Please enter a URL.")
        else:
            norm_url = normalize_url(url_input)
            
            # 1. Check Whitelist/Blacklist
            list_entry = list_collection.find_one({"url": norm_url})
            
            if list_entry:
                if list_entry['type'] == 'whitelist':
                    st.success(f"✅ **Legitimate** (Verified Whitelist)")
                    save_log(url_input, "Legitimate", 1.0, "Whitelist", True)
                else:
                    st.error(f"🚨 **Phishing** (Verified Blacklist)")
                    save_log(url_input, "Phishing", 1.0, "Blacklist", True)
            
            else:
                # 2. Use AI Model
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

# --- PAGE 2: Admin Dashboard ---
elif page == "Admin Dashboard":
    st.title("Admin Dashboard")
    
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Check secrets first, then env, then fallback
            admin_user = os.getenv("ADMIN_USER") or (st.secrets["ADMIN_USER"] if "ADMIN_USER" in st.secrets else "admin")
            admin_pass = os.getenv("ADMIN_PASS") or (st.secrets["ADMIN_PASS"] if "ADMIN_PASS" in st.secrets else "admin123")
            
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
        
        # --- Pending Reviews ---
        st.subheader("⚠️ Pending Reviews")
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
        st.subheader("📋 Recent Scan History")
        recent_logs = list(url_collection.find().sort("timestamp", -1).limit(20))
        
        if recent_logs:
            import pandas as pd
            df = pd.DataFrame(recent_logs)
            st.dataframe(df[['timestamp', 'url', 'status', 'confidence', 'source']], use_container_width=True)
        else:
            st.write("No history found.")