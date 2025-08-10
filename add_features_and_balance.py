import pandas as pd
import numpy as np
import torch
import soundfile as sf
import tempfile
import os
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertForSequenceClassification
from pyannote.audio import Model, Inference
from collections import Counter
import pickle

print("Loading models...")

# Load models
gender_model = Wav2Vec2ForSequenceClassification.from_pretrained("prithivMLmods/Common-Voice-Gender-Detection")
gender_processor = Wav2Vec2FeatureExtractor.from_pretrained("prithivMLmods/Common-Voice-Gender-Detection")
gender_model.eval()

tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-nano-zh")
sentiment_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
sentiment_model = BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')

snr_model = Model.from_pretrained("pyannote/brouhaha", 
                                  use_auth_token="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")
snr_inference = Inference(snr_model)

# Mappings
gender_map = {0: "female", 1: "male"}
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_gender(audio_array, sampling_rate):
    """Predict gender from audio"""
    try:
        inputs = gender_processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = gender_model(**inputs).logits
            pred_id = logits.argmax(dim=-1).item()
        return gender_map[pred_id]
    except Exception as e:
        return "unknown"

def get_sentiment(text):
    """Get sentiment from text"""
    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_id = torch.argmax(probs, dim=-1).item()
        return sentiment_map.get(sentiment_id, "unknown")
    except Exception as e:
        return "unknown"

def extract_snr_from_audio(audio_array, sampling_rate):
    """Extract SNR from audio"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio_array, sampling_rate)
        try:
            output = snr_inference(tmp_wav.name)
            snr_values = [snr for frame, (vad, snr, c50) in output]
            avg_snr = np.mean(snr_values) if snr_values else 0.0
        except Exception as e:
            avg_snr = 0.0
        finally:
            os.unlink(tmp_wav.name)
    return avg_snr

def add_features_to_data(combined_data):
    """Add features to the dataset"""
    print("Adding features...")
    
    combined_data_with_features = []
    
    for i, row in enumerate(tqdm(combined_data, desc="Processing")):
        audio, sampling_rate, label, text, audio_length, gender = row
        
        # Add tokens
        token_length = len(tokenizer.encode(text))
        
        # Add sentiment
        sentiment = get_sentiment(text)
        
        # Add gender prediction (if not available)
        predicted_gender = gender if gender is not None else predict_gender(audio, sampling_rate)
        
        # Add SNR
        snr = extract_snr_from_audio(audio, sampling_rate)
        
        # Create updated row
        updated_row = (audio, sampling_rate, label, text, audio_length, predicted_gender, token_length, sentiment, snr)
        combined_data_with_features.append(updated_row)
        
        # Print progress
        if i < 3 or i % 100 == 0:
            print(f"Item {i}: Gender={predicted_gender}, Sentiment={sentiment}, SNR={snr:.2f} dB")
    
    print(f"Added features to {len(combined_data_with_features)} samples")
    return combined_data_with_features

def create_balanced_dataset(combined_data_with_features, balance_features=['label', 'sentiment']):
    """Create balanced dataset"""
    print(f"Balancing by: {balance_features}")
    
    # Convert to DataFrame
    df = pd.DataFrame(combined_data_with_features, 
                     columns=["audio", "sampling_rate", "label", "text", "audio_length", "gender", "tokens", "sentiment", "snr"])
    
    # Get minimum count per combination
    feature_combinations = df.groupby(balance_features).size()
    min_count = feature_combinations.min()
    
    print(f"Minimum count per combination: {min_count}")
    
    # Sample equal numbers from each combination
    balanced_dfs = []
    for combo in feature_combinations.index:
        combo_df = df.copy()
        for i, feature in enumerate(balance_features):
            combo_df = combo_df[combo_df[feature] == combo[i]]
        
        if len(combo_df) > min_count:
            combo_df = combo_df.sample(n=min_count, random_state=42)
        
        balanced_dfs.append(combo_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Convert back to list of tuples
    balanced_data = []
    for _, row in balanced_df.iterrows():
        balanced_data.append((
            row['audio'], row['sampling_rate'], row['label'], row['text'], 
            row['audio_length'], row['gender'], row['tokens'], row['sentiment'], row['snr']
        ))
    
    print(f"Balanced dataset: {len(balanced_data)} samples")
    return balanced_data

def save_data(data, filename, include_audio=True):
    """Save data to file"""
    if include_audio:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} samples to {filename}")
    else:
        # Save metadata only
        metadata = []
        for audio, sampling_rate, label, text, audio_length, gender, tokens, sentiment, snr in data:
            metadata.append({
                'sampling_rate': sampling_rate, 'label': label, 'text': text,
                'audio_length': audio_length, 'gender': gender, 'tokens': tokens,
                'sentiment': sentiment, 'snr': snr
            })
        
        df = pd.DataFrame(metadata)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} samples to {filename}")

def main():
    """Main function"""
    
    # Load data from pickle
    print("Loading data from data/loaded_data_full.pkl...")
    try:
        with open('data/loaded_data_full.pkl', 'rb') as f:
            combined_data = pickle.load(f)
        print(f"Loaded {len(combined_data)} samples")
    except FileNotFoundError:
        print("data/loaded_data_full.pkl not found. Please run load_data.py first.")
        return
    
    # Show original distribution
    labels = [row[2] for row in combined_data]
    label_counts = Counter(labels)
    print("Original distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Add features
    combined_data_with_features = add_features_to_data(combined_data)
    
    # Save features data
    save_data(combined_data_with_features, 'data/features_data_full.pkl', include_audio=True)
    save_data(combined_data_with_features, 'data/features_data.csv', include_audio=False)
    
    # Show feature distributions
    print("\nFeature distributions:")
    sentiments = [row[7] for row in combined_data_with_features]
    genders = [row[5] for row in combined_data_with_features]
    snrs = [row[8] for row in combined_data_with_features]
    
    print("Sentiment:", Counter(sentiments))
    print("Gender:", Counter(genders))
    print("SNR range:", f"{min(snrs):.1f} - {max(snrs):.1f} dB")
    
    # Create balanced datasets
    print("\nCreating balanced datasets...")
    
    balanced_by_label = create_balanced_dataset(combined_data_with_features, ['label'])
    save_data(balanced_by_label, 'data/balanced_by_label_full.pkl', include_audio=True)
    
    balanced_by_sentiment = create_balanced_dataset(combined_data_with_features, ['sentiment'])
    save_data(balanced_by_sentiment, 'data/balanced_by_sentiment_full.pkl', include_audio=True)
    
    balanced_by_both = create_balanced_dataset(combined_data_with_features, ['label', 'sentiment'])
    save_data(balanced_by_both, 'data/balanced_data_full.pkl', include_audio=True)
    save_data(balanced_by_both, 'data/balanced_data.csv', include_audio=False)
    
    print("\nSummary:")
    print(f"Original: {len(combined_data)} samples")
    print(f"With features: {len(combined_data_with_features)} samples")
    print(f"Balanced by label: {len(balanced_by_label)} samples")
    print(f"Balanced by sentiment: {len(balanced_by_sentiment)} samples")
    print(f"Balanced by both: {len(balanced_by_both)} samples")

if __name__ == "__main__":
    main()
