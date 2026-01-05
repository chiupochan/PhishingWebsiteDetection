import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, LSTM, GRU, Bidirectional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.keras import TqdmCallback

# --- Configuration ---
MAX_LEN = 200
EMBEDDING_DIM = 32
EPOCHS = 10
BATCH_SIZE = 32
K_FOLDS = 5
MODELS_DIR = 'Models'
RESULTS_FILE = 'model_performance.csv'

def load_data(base_dir='Dataset'):
    print("Loading data...")
    try:
        X = np.load(os.path.join(base_dir, 'X_seq.npy'), allow_pickle=True)
        y = np.load(os.path.join(base_dir, 'y.npy'), allow_pickle=True)
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def get_vocab_size(X):
    return np.max(X) + 1

# --- Model Builders ---

def build_cnn(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="LSTM")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        GRU(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="GRU")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="CNN-BiLSTM")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(builder, model_name, X, y, vocab_size):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    print(f"\n--- Evaluating {model_name} ({K_FOLDS}-Fold CV) ---")
    
    best_fold_acc = -1
    
    fold_no = 1
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fresh model for each fold
        model = builder(vocab_size, MAX_LEN)
        
        print(f"Fold {fold_no}/{K_FOLDS}:")
        
        # Train with Progress Bar
        # verbose=0 suppresses the default Keras log
        # TqdmCallback(verbose=1) shows the progress bar
        model.fit(
            X_train, y_train, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            verbose=0, 
            callbacks=[TqdmCallback(verbose=1)]
        )
        
        # Predict
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['f1'].append(f1)
        
        print(f"Fold {fold_no} Result: Accuracy={acc:.4f}")
        
        # Save the BEST version of THIS model type
        if acc > best_fold_acc:
            best_fold_acc = acc
            save_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            model.save(save_path)
            
        fold_no += 1
        
    # Aggregate Results
    return {
        'Model': model_name,
        'Accuracy': np.mean(fold_metrics['accuracy']),
        'Accuracy_Std': np.std(fold_metrics['accuracy']),
        'Precision': np.mean(fold_metrics['precision']),
        'Precision_Std': np.std(fold_metrics['precision']),
        'Recall': np.mean(fold_metrics['recall']),
        'Recall_Std': np.std(fold_metrics['recall']),
        'F1_Score': np.mean(fold_metrics['f1']),
        'F1_Std': np.std(fold_metrics['f1'])
    }

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    X, y = load_data()
    if X is None: return
    
    vocab_size = get_vocab_size(X)
    print(f"Vocabulary Size: {vocab_size}")
    
    models = [
        (build_cnn, "CNN"),
        (build_lstm, "LSTM"),
        (build_gru, "GRU"),
        (build_cnn_bilstm, "CNN-BiLSTM")
    ]
    
    results = []
    
    for builder, name in models:
        try:
            res = train_and_evaluate(builder, name, X, y, vocab_size)
            results.append(res)
        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Results
    if results:
        df = pd.DataFrame(results)
        # Reorder for nice output
        cols = ['Model', 'Accuracy', 'Accuracy_Std', 'F1_Score', 'F1_Std', 
                'Precision', 'Precision_Std', 'Recall', 'Recall_Std']
        df = df[cols]
        
        csv_path = os.path.join(MODELS_DIR, RESULTS_FILE)
        df.to_csv(csv_path, index=False)
        
        print(f"\nAll models trained and saved to {MODELS_DIR}/")
        print(f"Detailed statistics saved to {csv_path}")
        print(df)

if __name__ == "__main__":
    main()