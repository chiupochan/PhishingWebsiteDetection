import os
import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, 
                                     Bidirectional, LSTM, GRU, Dense, Dropout, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration Settings ---
DATA_PATH = "Dataset/phishing_site_urls.csv"
MODEL_SAVE_PATH = "Models/CNN-BiLSTM_best.keras"
TOKENIZER_SAVE_PATH = "tokenizer.pkl"
RESULTS_DIR = "Results"

MAX_LEN = 100               
EMBEDDING_DIM = 32          
EPOCHS = 20                 
BATCH_SIZE = 128             
K_FOLDS = 5                 
RANDOM_STATE = 42

# Ensure directories exist
os.makedirs("Models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def clean_url(url):
    """Normalizes URL by lowercasing and stripping common prefixes."""
    url = str(url).lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url

def load_and_preprocess_data(filepath):
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(filepath)
    df.dropna(subset=['URL', 'Label'], inplace=True)
    df.drop_duplicates(subset=['URL'], inplace=True)
    df['URL'] = df['URL'].apply(clean_url)
    df['Label'] = df['Label'].map({'bad': 1, 'good': 0})
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df['URL'].values, df['Label'].values

# --- Model Builder Functions ---

def build_cnn_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm_model(vocab_size):
    """Fine-tuned Hybrid Architecture for peak performance"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Visualization Helpers ---

def plot_and_save_history(histories, model_name):
    """Averages history across folds and plots accuracy vs epochs"""
    mean_acc = np.mean([h['accuracy'] for h in histories], axis=0)
    mean_val_acc = np.mean([h['val_accuracy'] for h in histories], axis=0)
    
    plt.figure(figsize=(8, 5))
    plt.plot(mean_acc, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(mean_val_acc, label='Validation Accuracy', color='orange', linewidth=2)
    plt.title(f'{model_name}: Average Accuracy vs Epochs (5-Fold CV)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, f'accuracy_vs_epoch_{model_name}.png'))
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, model_name):
    """Plots confusion matrix from Out-Of-Fold predictions"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png'))
    plt.close()

def main():
    # 1. Load and Preprocess
    X_raw, y = load_and_preprocess_data(DATA_PATH)
    
    # 2. Sequence Vectorization
    print("Tokenizing characters...")
    tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_raw)
    vocab_size = len(tokenizer.word_index) + 1
    
    X_seq = tokenizer.texts_to_sequences(X_raw)
    X = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    architectures = {
        'CNN': build_cnn_model,
        'LSTM': build_lstm_model,
        'GRU': build_gru_model,
        'CNN-BiLSTM': build_cnn_bilstm_model
    }
    
    # Tracking structures
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    final_metrics = {model: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []} for model in architectures}
    fold_histories = {model: [] for model in architectures}
    oof_predictions = {model: np.zeros(len(y)) for model in architectures}
    
    print(f"\nStarting {K_FOLDS}-Fold Stratified Cross-Validation...")
    
    # 3. K-Fold Training Loop
    fold_iterator = tqdm(enumerate(skf.split(X, y), 1), total=K_FOLDS, desc="Cross-Validation Progress")
    
    for fold, (train_idx, val_idx) in fold_iterator:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for model_name, builder_func in architectures.items():
            model = builder_func(vocab_size)
            
            # Train and track history
            history = model.fit(X_train, y_train, 
                                validation_data=(X_val, y_val),
                                epochs=EPOCHS, 
                                batch_size=BATCH_SIZE, 
                                verbose=1) 
            fold_histories[model_name].append(history.history)
            
            # Predict and evaluate
            y_pred_prob = model.predict(X_val, verbose=0)
            y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
            
            # Save Out-Of-Fold predictions for the confusion matrix
            oof_predictions[model_name][val_idx] = y_pred
            
            # Append fold metrics
            final_metrics[model_name]['accuracy'].append(accuracy_score(y_val, y_pred))
            final_metrics[model_name]['precision'].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
            final_metrics[model_name]['recall'].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            final_metrics[model_name]['roc_auc'].append(roc_auc_score(y_val, y_pred_prob))

    # 4. Generate Visualizations and Metrics Table
    print("\nGenerating Visualizations and Summary Table...")
    metrics_summary = []
    
    for model_name in architectures.keys():
        # Generate Plots
        plot_and_save_history(fold_histories[model_name], model_name)
        plot_and_save_confusion_matrix(y, oof_predictions[model_name], model_name)
        
        # Aggregate Metrics
        mean_metrics = {
            'Model': model_name,
            'Accuracy': np.mean(final_metrics[model_name]['accuracy']),
            'Precision': np.mean(final_metrics[model_name]['precision']),
            'Recall': np.mean(final_metrics[model_name]['recall']),
            'F1-Score': np.mean(final_metrics[model_name]['f1']),
            'ROC-AUC': np.mean(final_metrics[model_name]['roc_auc'])
        }
        metrics_summary.append(mean_metrics)
    
    # Save and display DataFrame
    df_metrics = pd.DataFrame(metrics_summary).set_index('Model')
    df_metrics.to_csv(os.path.join(RESULTS_DIR, 'metrics_summary.csv'))
    print("\n=== Comparative Cross-Validation Results ===")
    print(df_metrics.to_string(float_format="{:.4f}".format))
    
    # 5. Train Final Global Model (CNN-BiLSTM) and Export
    print("\nTraining final fine-tuned CNN-BiLSTM model on full dataset for deployment...")
    final_model = build_cnn_bilstm_model(vocab_size)
    
    # Callbacks for final training
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    final_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks)
    
    final_model.save(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print(f"\nDeployment assets saved:\n- Model: {MODEL_SAVE_PATH}\n- Tokenizer: {TOKENIZER_SAVE_PATH}")
    print(f"- Visualizations & Metrics: ./{RESULTS_DIR}/")

if __name__ == "__main__":
    main()