import pandas as pd
import os
import re

def load_data(filepath):
    try:
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            print("Please ensure your file is named 'kaggle_dataset.csv' and placed in the Dataset/ folder.")
            return None
            
        # Try reading with default comma separator
        df = pd.read_csv(filepath)
        
        # Check if it loaded correctly (sometimes Kaggle datasets use different separators)
        if len(df.columns) < 2:
            print("Warning: CSV might not be comma-separated. Trying tab separator...")
            df = pd.read_csv(filepath, sep='\t')
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def normalize_url(url):
    """
    Removes http://, https://, and www. to standardize the URL format.
    """
    if not isinstance(url, str):
        return ""
    
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    
    return url

def preprocess_data(df, max_samples=50000):
    print("--- Preprocessing Kaggle Dataset ---")
    
    # 1. Standardize Column Names
    df.columns = [c.lower().strip() for c in df.columns]
    
    if 'url' not in df.columns or 'label' not in df.columns:
        print(f"Error: Expected columns 'url' and 'label'. Found: {df.columns}")
        return None
        
    # 2. Invert Labels (Based on your sample: 0=Phishing, 1=Legit)
    # New Label: 1 if old was 0, 0 if old was 1.
    print("Inverting labels to standard format (1=Phishing, 0=Legitimate)...")
    df['label'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    # 3. Normalize URLs
    print("Normalizing URL formats (removing http/s and www)...")
    df['url'] = df['url'].apply(normalize_url)
    
    # 4. Cleaning
    initial_count = len(df)
    df.dropna(subset=['url'], inplace=True)
    df = df[df['url'] != '']
    df.drop_duplicates(subset=['url'], inplace=True)
    print(f"Removed {initial_count - len(df)} duplicates/empty rows.")
    
    # 5. Check Class Balance & Limit Data
    counts = df['label'].value_counts()
    print(f"\nClass Distribution (After Inversion):\n{counts}")
    
    n_phishing = counts.get(1, 0)
    n_legitimate = counts.get(0, 0)
    
    # Cap the data at max_samples per class
    target_count = min(n_phishing, n_legitimate, max_samples)
    
    if target_count > 0:
        print(f"\nBalancing and Limiting data. Target samples per class: {target_count}")
        
        phishing_df = df[df['label'] == 1].sample(n=target_count, random_state=42)
        legitimate_df = df[df['label'] == 0].sample(n=target_count, random_state=42)
        
        balanced_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
        
        # Shuffle
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return balanced_df
    else:
        print("Error: One class has 0 samples.")
        return None

def main():
    base_dir = 'Dataset'
    input_path = os.path.join(base_dir, 'kaggle_dataset.csv')
    output_path = os.path.join(base_dir, 'processed_data.csv')
    
    print(f"Looking for data at {input_path}...")
    
    df = load_data(input_path)
    
    if df is not None:
        # Pass the 50,000 limit here
        processed_df = preprocess_data(df, max_samples=50000)
        
        if processed_df is not None:
            print(f"Saving processed data to {output_path}...")
            processed_df.to_csv(output_path, index=False)
            print("Done.")
            print(f"Total samples: {len(processed_df)}")
            print("First 5 rows:")
            print(processed_df.head())

if __name__ == "__main__":
    main()