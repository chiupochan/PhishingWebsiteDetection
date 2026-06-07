import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional, LSTM, GRU, Dense, Dropout

# --- Configuration Settings ---
DATA_PATH = "phishing_site_urls.csv"
MODEL_SAVE_PATH = "cnn_bilstm_model.h5"
TOKENIZER_SAVE_PATH = "tokenizer.pkl"

MAX_LEN = 100               # Max sequence length
EMBEDDING_DIM = 32          # Embedding dimensions 
EPOCHS = 10                 # Number of epochs
BATCH_SIZE = 64             # Batch size 
K_FOLDS = 5                 # Stratified K-Fold splits 
RANDOM_STATE = 42

def clean_url(url):
    """Normalizes URL by lowercasing and stripping common prefixes."""
    url = str(url).lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url

def load_and_preprocess_data(filepath):
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(filepath)
    
    # Data Cleaning: Remove nulls and duplicates
    df.dropna(subset=['URL', 'Label'], inplace=True)
    df.drop_duplicates(subset=['URL'], inplace=True)
    
    # Text Normalization
    df['URL'] = df['URL'].apply(clean_url)
    
    # Label Standardization: Inversion (Phishing=1, Legitimate=0)
    df['Label'] = df['Label'].map({'bad': 1, 'good': 0})
    
    # Data Shuffling
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    return df['URL'].values, df['Label'].values

# --- Model Builder Functions ---

def build_cnn_model(vocab_size):
    """Constructs the baseline CNN architecture"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5), # Optimal CNN dropout
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_size):
    """Constructs the baseline LSTM architecture"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(64), # Optimal LSTM units
        Dense(64, activation='relu'),
        Dropout(0.3), # Optimal LSTM dropout
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(vocab_size):
    """Constructs the baseline GRU architecture"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(64), # Optimal GRU units
        Dense(64, activation='relu'),
        Dropout(0.3), # Optimal GRU dropout
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm_model(vocab_size):
    """Constructs the hybrid CNN-BiLSTM architecture"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(32)), # Optimal CNN-BiLSTM units
        Dense(64, activation='relu'),
        Dropout(0.3), # Optimal CNN-BiLSTM dropout
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Load and Preprocess
    X_raw, y = load_and_preprocess_data(DATA_PATH)
    
    # 2. Sequence Vectorization (Character-level Tokenization)
    print("Tokenizing characters...")
    tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_raw)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    # Convert and pad sequences (Post-padding & Post-truncating)
    X_seq = tokenizer.texts_to_sequences(X_raw)
    X = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Define models to evaluate
    architectures = {
        'CNN': build_cnn_model,
        'LSTM': build_lstm_model,
        'GRU': build_gru_model,
        'CNN-BiLSTM': build_cnn_bilstm_model
    }
    
    # 3. Stratified 5-Fold Cross-Validation for ALL models
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Dictionary to store final averaged metrics for each architecture
    final_results = {model_name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []} 
                     for model_name in architectures.keys()}
    
    print(f"Starting {K_FOLDS}-Fold Stratified Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Processing Fold {fold} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for model_name, builder_func in architectures.items():
            print(f"Training {model_name}...")
            model = builder_func(vocab_size)
            
            # Train Model
            model.fit(X_train, y_train, 
                      validation_data=(X_val, y_val),
                      epochs=EPOCHS, 
                      batch_size=BATCH_SIZE, 
                      verbose=0) # Set to 1 if you want to see epoch progress
            
            # Evaluate Model
            y_pred_prob = model.predict(X_val, verbose=0)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Macro-averaged metrics
            final_results[model_name]['accuracy'].append(accuracy_score(y_val, y_pred))
            final_results[model_name]['precision'].append(precision_score(y_val, y_pred, average='macro'))
            final_results[model_name]['recall'].append(recall_score(y_val, y_pred, average='macro'))
            final_results[model_name]['f1'].append(f1_score(y_val, y_pred, average='macro'))
            final_results[model_name]['roc_auc'].append(roc_auc_score(y_val, y_pred_prob))
    
    # 4. Display Comparative Cross-Validation Results
    print("\n=== Comparative Cross-Validation Results (Averages) ===")
    for model_name, metrics in final_results.items():
        print(f"\n{model_name} Performance:")
        for metric, values in metrics.items():
            print(f"  Mean {metric.capitalize()}: {np.mean(values):.4f}")
    
    # 5. Train Final Global Model (CNN-BiLSTM) and Export
    print("\nTraining final CNN-BiLSTM model on full dataset for deployment...")
    final_model = build_cnn_bilstm_model(vocab_size)
    final_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    final_model.save(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print(f"\nDeployment assets saved:\n- Model: {MODEL_SAVE_PATH}\n- Tokenizer: {TOKENIZER_SAVE_PATH}")

if __name__ == "__main__":
    main()