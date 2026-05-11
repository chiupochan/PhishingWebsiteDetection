import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration & Directories ---
MAX_LEN = 100 
K_FOLDS = 5   
EPOCHS = 10   

# Output Directories
MODELS_DIR = 'Models'
RESULTS_DIR = 'Results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'Plots')
HISTORIES_DIR = os.path.join(RESULTS_DIR, 'Histories')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'Excel_Reports')

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, HISTORIES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Hyperparameter Grids ---
# I added multiple values here so your "Range" column in the final report 
# actually shows a search space (e.g., [32, 64]). 
# Note: More values = longer training time.
PARAM_GRIDS = {
    "CNN": {'embedding_dim': [32], 'filters': [32, 64], 'dense_units': [64], 'dropout': [0.3, 0.5], 'batch_size': [64]},
    "LSTM": {'embedding_dim': [32], 'lstm_units': [32, 64], 'dense_units': [64], 'dropout': [0.3, 0.5], 'batch_size': [64]},
    "GRU": {'embedding_dim': [32], 'gru_units': [32, 64], 'dense_units': [64], 'dropout': [0.3, 0.5], 'batch_size': [64]},
    "CNN-BiLSTM": {'embedding_dim': [32], 'filters': [64], 'lstm_units': [32, 64], 'dense_units': [64], 'dropout': [0.3, 0.5], 'batch_size': [64]}
}

def load_data(base_dir='Dataset'):
    print("Loading data...")
    try:
        X = np.load(os.path.join(base_dir, 'X_seq.npy'), allow_pickle=True)
        y = np.load(os.path.join(base_dir, 'y.npy'), allow_pickle=True)
        if X.shape[1] > MAX_LEN:
            X = X[:, :MAX_LEN]
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def get_vocab_size(X):
    return np.max(X) + 1

# --- Model Builders ---
def build_cnn(vocab_size, max_len, p):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=p['embedding_dim']),
        Conv1D(filters=p['filters'], kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(p['dense_units'], activation='relu'),
        Dropout(p['dropout']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(vocab_size, max_len, p):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=p['embedding_dim']),
        LSTM(p['lstm_units']),
        Dense(p['dense_units'], activation='relu'),
        Dropout(p['dropout']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru(vocab_size, max_len, p):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=p['embedding_dim']),
        GRU(p['gru_units']),
        Dense(p['dense_units'], activation='relu'),
        Dropout(p['dropout']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm(vocab_size, max_len, p):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=p['embedding_dim']),
        Conv1D(filters=p['filters'], kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(p['lstm_units'])),
        Dense(p['dense_units'], activation='relu'),
        Dropout(p['dropout']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Utility Functions ---
def plot_and_save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate (0)', 'Phishing (1)'],
                yticklabels=['Legitimate (0)', 'Phishing (1)'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()

def plot_model_comparison(df_results):
    best_models_df = df_results.loc[df_results.groupby('Model')['Accuracy'].idxmax()]
    melted_df = best_models_df.melt(
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Macro_F1', 'Weighted_F1'], 
        var_name='Metric', 
        value_name='Score'
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Best Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.legend(loc='lower right', title='Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_performance_comparison.png'), dpi=300)
    plt.close()

def save_training_history(histories, model_name, param_idx):
    avg_history = {}
    for key in histories[0].keys():
        min_epochs = min([len(h[key]) for h in histories])
        avg_history[key] = np.mean([h[key][:min_epochs] for h in histories], axis=0)
    
    df_hist = pd.DataFrame(avg_history)
    df_hist.index.name = 'Epoch'
    df_hist.to_csv(os.path.join(HISTORIES_DIR, f"{model_name}_params_v{param_idx}_history.csv"))

# --- Main Training Loop ---
def train_and_evaluate_grid(builder, model_name, X, y, vocab_size, param_grid):
    grid = list(ParameterGrid(param_grid))
    print(f"\n{'='*50}\nStarting {model_name} ({len(grid)} configurations)\n{'='*50}")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    all_results = []
    best_overall_acc = -1
    best_fold_model = None
    
    for idx, params in enumerate(grid):
        print(f"\n[{model_name}] Config {idx+1}: {params}")
        
        fold_histories, oof_y_true, oof_y_pred, oof_y_prob = [], [], [], []
        best_fold_acc = -1
        
        for fold_no, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  -> Fold {fold_no}/{K_FOLDS}...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = builder(vocab_size, MAX_LEN, params)
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=EPOCHS, batch_size=params['batch_size'], 
                verbose=1, callbacks=[early_stop] 
            )
            fold_histories.append(history.history)
            
            y_pred_prob = model.predict(X_val, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            oof_y_true.extend(y_val)
            oof_y_pred.extend(y_pred)
            oof_y_prob.extend(y_pred_prob)
            
            fold_acc = accuracy_score(y_val, y_pred)
            if fold_acc > best_fold_acc:
                best_fold_acc = fold_acc
                best_fold_model = model
            
            tf.keras.backend.clear_session()
                
        acc = accuracy_score(oof_y_true, oof_y_pred)
        auc = roc_auc_score(oof_y_true, oof_y_prob)
        
        report_dict = classification_report(oof_y_true, oof_y_pred, target_names=['Legitimate', 'Phishing'], output_dict=True)
        
        macro_f1 = report_dict['macro avg']['f1-score']
        weighted_f1 = report_dict['weighted avg']['f1-score']
        
        print(f"Result -> Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
        
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_excel(os.path.join(REPORTS_DIR, f"{model_name}_config_{idx+1}_report.xlsx"))
        
        save_training_history(fold_histories, model_name, idx+1)
        
        result_row = {
            'Model': model_name,
            'Config_ID': idx+1,
            'Accuracy': acc, 
            'ROC_AUC': auc,
            'Macro_F1': macro_f1,
            'Weighted_F1': weighted_f1,
            'Macro_Precision': report_dict['macro avg']['precision'],
            'Weighted_Precision': report_dict['weighted avg']['precision'],
            'Macro_Recall': report_dict['macro avg']['recall'],
            'Weighted_Recall': report_dict['weighted avg']['recall'],
            **params 
        }
        all_results.append(result_row)
        
        plot_and_save_confusion_matrix(
            oof_y_true, oof_y_pred, 
            f"{model_name} Confusion Matrix", 
            f"{model_name}_config_{idx+1}_cm.png"
        )
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            if best_fold_model is not None:
                best_fold_model.save(os.path.join(MODELS_DIR, f"{model_name}_best.h5"))

    return all_results

def main():
    X, y = load_data()
    if X is None: return
    
    vocab_size = get_vocab_size(X)
    print(f"Vocabulary Size: {vocab_size}")
    
    models = [
        (build_cnn, "CNN", PARAM_GRIDS["CNN"]),
        (build_lstm, "LSTM", PARAM_GRIDS["LSTM"]),
        (build_gru, "GRU", PARAM_GRIDS["GRU"]),
        (build_cnn_bilstm, "CNN-BiLSTM", PARAM_GRIDS["CNN-BiLSTM"])
    ]
    
    all_grid_results = []
    
    for builder, name, param_grid in models:
        try:
            results = train_and_evaluate_grid(builder, name, X, y, vocab_size, param_grid)
            all_grid_results.extend(results)
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    # --- Data Aggregation & Output Generation ---
    if all_grid_results:
        df_results = pd.DataFrame(all_grid_results)
        # Sort by Macro_F1 to easily identify the top performing configurations
        df_results = df_results.sort_values(by=['Model', 'Macro_F1'], ascending=[True, False])
        
        # 1. Save Master Results
        csv_path = os.path.join(RESULTS_DIR, 'master_hyperparameter_results.csv')
        df_results.to_csv(csv_path, index=False)
        
        # 2. Extract Optimal Hyperparameters and Build the Requested Table
        best_models_df = df_results.loc[df_results.groupby('Model')['Macro_F1'].idxmax()]
        
        hyperparam_summary = []
        for model_name, grid in PARAM_GRIDS.items():
            # Get the row of the best performing config for this specific model
            best_row = best_models_df[best_models_df['Model'] == model_name].iloc[0]
            
            for param_name, param_range in grid.items():
                hyperparam_summary.append({
                    'Model': model_name,
                    'Hyperparameter': param_name,
                    'Range': str(param_range), # Formats the list (e.g., [32, 64]) cleanly into text
                    'Optimal Value': best_row[param_name]
                })
        
        df_hyperparams = pd.DataFrame(hyperparam_summary)
        hyperparams_path = os.path.join(REPORTS_DIR, 'optimal_hyperparameters.xlsx')
        df_hyperparams.to_excel(hyperparams_path, index=False)
        
        # 3. Create Model Comparison Bar Chart
        plot_model_comparison(df_results)
        
        print("\n" + "="*50)
        print("ALL TRAINING AND EVALUATION COMPLETE!")
        print("="*50)
        print(f"- Optimal Hyperparameters Table: {hyperparams_path}")
        print(f"- Confusion Matrices & Bar Chart: {PLOTS_DIR}/")
        print(f"- Excel Classification Reports: {REPORTS_DIR}/")
        print(f"- Master Data CSV: {csv_path}")

if __name__ == "__main__":
    main()