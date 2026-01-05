import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, 
                                     Flatten, Dropout, Bidirectional)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. DATA PREPARATION ---
def load_and_preprocess():
    # Load structured datasets
    legit_df = pd.read_csv("Dataset/structured_legitimate_list.csv")
    phish_df = pd.read_csv("Dataset/structured_phishing_list.csv")
    df = pd.concat([legit_df, phish_df], ignore_index=True)
    
    # --- REMOVE BIASED FEATURES ---
    # We drop non-features and the features causing 99% accuracy
    features_to_drop = ['URL', 'label', 'dir_count', 'url_len']
    X = df.drop(features_to_drop, axis=1).values
    y = df['label'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for sequential models: (samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, X.shape[1]

# --- 2. MODEL FACTORIES ---

def get_cnn(dim):
    return Sequential([
        Input(shape=(dim, 1)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

def get_lstm(dim):
    return Sequential([
        Input(shape=(dim, 1)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

def get_gru(dim):
    return Sequential([
        Input(shape=(dim, 1)),
        GRU(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])


def get_cnn_bilstm(dim):
    return Sequential([
        Input(shape=(dim, 1)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Bidirectional(LSTM(64)),
        Dense(1, activation='sigmoid')
    ])

# --- 3. CROSS-VALIDATION LOOP ---
X, y, input_dim = load_and_preprocess()
print(f"Training on {input_dim} features (excluding biased URL metrics).")

model_factories = {
    "cnn": get_cnn, "lstm": get_lstm, "gru": get_gru, "cnn_bilstm": get_cnn_bilstm
}

os.makedirs("Models", exist_ok=True)
final_results = {}

for name, factory in model_factories.items():
    print(f"\n--- 10-Fold CV: {name.upper()} ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = factory(input_dim)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        _, acc = model.evaluate(X_val, y_val, verbose=0)
        fold_accs.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
    
    avg_acc = np.mean(fold_accs)
    final_results[name] = avg_acc
    print(f"Average Accuracy for {name}: {avg_acc:.4f}")
    
    # Save the model
    model.save(f"Models/{name}_model.keras")

print("\n--- Final Results ---")
for m_name, accuracy in final_results.items():
    print(f"{m_name}: {accuracy:.4f}")