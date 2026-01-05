import pandas as pd
import numpy as np
import os
import pickle

# --- Configuration ---
# Max length of URL to consider (URLs longer than this are truncated)
# 200 is usually sufficient for phishing detection
MAX_LEN = 200 

def load_data(filepath):
    """Loads the processed CSV data."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    return pd.read_csv(filepath)

def create_char_tokenizer(texts):
    """
    Creates a dictionary mapping unique characters to integers.
    Reserved: 0 is for padding.
    """
    # Get all unique characters from the dataset
    unique_chars = set(''.join(texts.astype(str)))
    
    # Ensure common URL characters are included even if not in the sample
    common_chars = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
    unique_chars.update(common_chars)
    
    # Create mapping: char -> int (start from 1, 0 is padding)
    sorted_chars = sorted(list(unique_chars))
    char_to_int = {c: i + 1 for i, c in enumerate(sorted_chars)}
    
    print(f"Vocabulary Size: {len(char_to_int)} unique characters.")
    return char_to_int

def tokenize_and_pad(texts, char_to_int, max_len):
    """
    Converts URLs to integer sequences and pads/truncates them.
    """
    sequences = []
    for text in texts:
        # Convert char to int, ignore unknown chars
        seq = [char_to_int.get(c, 0) for c in str(text)] 
        sequences.append(seq)
    
    # Create a zero-filled matrix (Padding)
    data_matrix = np.zeros((len(sequences), max_len), dtype='int32')
    
    for i, seq in enumerate(sequences):
        if len(seq) > max_len:
            # Truncate if too long
            data_matrix[i] = seq[:max_len]
        else:
            # Insert sequence (Post-padding happens naturally as the rest remains 0)
            data_matrix[i, :len(seq)] = seq
            
    return data_matrix

def extract_lexical_features(urls):
    """
    Extracts hand-crafted statistical features.
    Useful for Hybrid models or baseline comparisons.
    """
    features = []
    for url in urls:
        url = str(url)
        row = [
            len(url),                   # Total length
            url.count('.'),             # Dot count
            url.count('-'),             # Hyphen count
            url.count('@'),             # At symbol count
            url.count('/'),             # Slash count
            url.count('?'),             # Question mark count
            url.count('='),             # Equals count
            sum(c.isdigit() for c in url), # Digit count
            sum(c.isalpha() for c in url)  # Letter count
        ]
        features.append(row)
    return np.array(features)

def main():
    base_dir = 'Dataset'
    input_path = os.path.join(base_dir, 'processed_data.csv')
    
    # Output paths
    output_x_seq = os.path.join(base_dir, 'X_seq.npy')     # Sequence data (for CNN/LSTM)
    output_x_lex = os.path.join(base_dir, 'X_lex.npy')     # Lexical features (for Hybrid/Dense)
    output_y = os.path.join(base_dir, 'y.npy')             # Labels
    output_tokenizer = os.path.join(base_dir, 'tokenizer.pkl') # Saved tokenizer
    
    print("--- Starting Feature Engineering ---")
    
    df = load_data(input_path)
    if df is None:
        return

    urls = df['url'].values
    labels = df['label'].values
    
    # 1. Tokenization (The primary feature for Deep Learning)
    print("Creating Tokenizer...")
    char_to_int = create_char_tokenizer(urls)
    
    print(f"Tokenizing and Padding sequences (Max Len: {MAX_LEN})...")
    X_seq = tokenize_and_pad(urls, char_to_int, MAX_LEN)
    
    # 2. Lexical Feature Extraction (Optional/Auxiliary)
    print("Extracting Lexical Features (Length, counts, etc.)...")
    X_lex = extract_lexical_features(urls)
    
    # 3. Saving
    print("Saving features to 'Dataset/'...")
    np.save(output_x_seq, X_seq)
    np.save(output_x_lex, X_lex)
    np.save(output_y, labels)
    
    with open(output_tokenizer, 'wb') as f:
        pickle.dump(char_to_int, f)
        
    print("--- Feature Engineering Completed ---")
    print(f"Sequence Data Shape (X_seq): {X_seq.shape}")
    print(f"Lexical Data Shape  (X_lex): {X_lex.shape}")
    print(f"Labels Shape        (y):     {labels.shape}")
    print(f"Files saved: X_seq.npy, X_lex.npy, y.npy, tokenizer.pkl")

if __name__ == "__main__":
    main()