from datasets import load_dataset
import pandas as pd
import pickle
from tqdm import tqdm

def process_cantonese(cantonese_corpus, max_samples=100):
    """Process Cantonese data with audio arrays"""
    if cantonese_corpus is None:
        print("Cantonese corpus not available, skipping...")
        return []
    
    data = []
    try:
        for i, row in enumerate(tqdm(cantonese_corpus, desc="Processing Cantonese", total=max_samples)):
            audio = row['audio']['array']
            sampling_rate = row['audio']['sampling_rate']
            label = 'cantonese'
            text = row.get('sentence', '')
            audio_length = len(audio) / sampling_rate
            gender = row.get('gender', None)
            data.append((audio, sampling_rate, label, text, audio_length, gender))
            if i >= max_samples - 1:
                break
    except Exception as e:
        print(f"Error processing Cantonese corpus: {e}")
        print(f"Processed {len(data)} Cantonese samples before error")
    
    return data

def main():
    # Load Cantonese corpus
    print("Loading Cantonese corpus...")
    try:
        cantonese_corpus = load_dataset("ziyou-li/cantonese_daily", split="train", streaming=True)
        print("Cantonese corpus loaded successfully")
    except Exception as e:
        print(f"Error loading Cantonese corpus: {e}")
        return
    
    # Process Cantonese data
    print("=== Processing Cantonese Dataset ===")
    cantonese_data = process_cantonese(cantonese_corpus)
    print(f"Cantonese: {len(cantonese_data)} samples")
    
    # Update CSV file
    print("Updating CSV file...")
    try:
        existing_df = pd.read_csv('data/loaded_data.csv')
        print(f"Loaded {len(existing_df)} existing samples from CSV")
        
        # Add Cantonese metadata to CSV
        cantonese_metadata = []
        for audio, sampling_rate, label, text, audio_length, gender in cantonese_data:
            cantonese_metadata.append({
                'sampling_rate': sampling_rate,
                'label': label,
                'text': text,
                'audio_length': audio_length,
                'gender': gender,
                'audio_shape': audio.shape if hasattr(audio, 'shape') else 'No shape'
            })
        
        cantonese_df = pd.DataFrame(cantonese_metadata)
        combined_df = pd.concat([existing_df, cantonese_df], ignore_index=True)
        combined_df.to_csv('data/loaded_data.csv', index=False)
        print(f"Updated CSV: {len(combined_df)} total samples")
        
    except FileNotFoundError:
        print("data/loaded_data.csv not found. Creating new file with Cantonese data only.")
        cantonese_metadata = []
        for audio, sampling_rate, label, text, audio_length, gender in cantonese_data:
            cantonese_metadata.append({
                'sampling_rate': sampling_rate,
                'label': label,
                'text': text,
                'audio_length': audio_length,
                'gender': gender,
                'audio_shape': audio.shape if hasattr(audio, 'shape') else 'No shape'
            })
        
        combined_df = pd.DataFrame(cantonese_metadata)
        combined_df.to_csv('data/loaded_data.csv', index=False)
        print(f"Created CSV: {len(combined_df)} samples")
    
    # Update pickle file
    print("Updating pickle file...")
    try:
        with open('data/loaded_data_full.pkl', 'rb') as f:
            existing_data = pickle.load(f)
        print(f"Loaded {len(existing_data)} existing samples from pickle")
        
        # Combine with Cantonese data
        combined_data = existing_data + cantonese_data
        
    except FileNotFoundError:
        print("data/loaded_data_full.pkl not found. Creating new file with Cantonese data only.")
        combined_data = cantonese_data
    
    # Save updated pickle file
    with open('data/loaded_data_full.pkl', 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"Updated pickle: {len(combined_data)} total samples")
    
    # Show final distribution
    print("\n=== Final Dataset Summary ===")
    labels = [row[2] for row in combined_data]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    print(f"\nTotal samples: {len(combined_data)}")
    print("Successfully updated both CSV and pickle files!")

if __name__ == "__main__":
    main()
