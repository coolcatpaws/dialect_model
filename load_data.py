from datasets import load_dataset, Audio
import torch
from tqdm import tqdm
import pandas as pd
import time
import os
import pickle

torch.multiprocessing.set_sharing_strategy('file_system') # This is necessary to avoid issues with multiprocessing in PyTorch

# Load datasets with error handling
try:
    print("Loading Shanghai corpus...")
    shanghai_corpus = load_dataset("TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus", split = "train")
    print("Shanghai corpus loaded successfully")
except Exception as e:
    print(f"Error loading Shanghai corpus: {e}")
    shanghai_corpus = None

try:
    print("Loading Mandarin corpus...")
    mandarin_corpus = load_dataset("urarik/free_st_chinese_mandarin_corpus", split="train", streaming=True)
    print("Mandarin corpus loaded successfully")
except Exception as e:
    print(f"Error loading Mandarin corpus: {e}")
    mandarin_corpus = None

try:
    print("Loading Sichuan corpus...")
    sichuan_corpus = load_dataset("wanghaikuan/sichuan", split="train", streaming=True) #6k rows 
    print("Sichuan corpus loaded successfully")
except Exception as e:
    print(f"Error loading Sichuan corpus: {e}")
    sichuan_corpus = None

try:
    print("Loading Cantonese corpus...")
    cantonese_corpus = load_dataset("ziyou-li/cantonese_daily", split="train", streaming=True)
    print("Cantonese corpus loaded successfully")
except Exception as e:
    print(f"Error loading Cantonese corpus: {e}")
    cantonese_corpus = None

# note: streaming=True prevents a download that exceeds my computer space limit but can load full on in container 

# from datasets import config
# print(config.HF_DATASETS_CACHE) #to see where the datasets are stored

#pulls out audio, sample, transcription, and label from datasets and organizes them into a list of tuples
def process_shanghai(shanghai_corpus, max_samples=3000):
    if shanghai_corpus is None:
        print("Shanghai corpus not available, skipping...")
        return []
    
    data = []
    for i, row in enumerate(shanghai_corpus):
        audio = row['audio']['array']
        sampling_rate = row['audio']['sampling_rate']
        label = 'shanghai'  # Label for Shanghai dataset
        text = row.get('transcription', '')  # Get transcription if available
        audio_length = len(audio) / sampling_rate  # Calculate audio length in seconds
        gender = row.get('gender', None)  # Get gender if available
        data.append((audio, sampling_rate, label, text, audio_length, gender))
        if i >= max_samples - 1:
            break
    return data

def process_mandarin(mandarin_corpus, max_samples=3000):
    if mandarin_corpus is None:
        print("Mandarin corpus not available, skipping...")
        return []
    
    data = []
    for i, row in enumerate(mandarin_corpus):
        audio = row['audio']['array']
        sampling_rate = row['audio']['sampling_rate']
        label = 'mandarin'  # Label for Mandarin dataset
        text = row.get('sentence', '')  # Get sentence if available
        audio_length = len(audio) / sampling_rate  # Calculate audio length in seconds
        gender = row.get('gender', None)  # Get gender if available
        data.append((audio, sampling_rate, label, text, audio_length, gender))
        if i >= max_samples - 1:
            break
    return data


# Process Sichuan and Cantonese datasets with progress bars since they are streaming 
# Takes incredibly long to process the full datasets, so we limit the number of samples processed for demo

def process_sichuan(sichuan_corpus, max_samples=100):
    if sichuan_corpus is None:
        print("Sichuan corpus not available, skipping...")
        return []
    
    data = []
    try:
        for i, row in enumerate(tqdm(sichuan_corpus, desc="Processing Sichuan", total=max_samples)):
            audio = row['audio']['array']
            sampling_rate = row['audio']['sampling_rate']
            label = 'sichuan'
            text = row.get('sentence', '')
            audio_length = len(audio) / sampling_rate  # Calculate audio length in seconds
            gender = row.get('gender', None)  # Get gender if available
            data.append((audio, sampling_rate, label, text, audio_length, gender))
            if i >= max_samples - 1:
                break
    except Exception as e:
        print(f"Error processing Sichuan corpus: {e}")
        print(f"Processed {len(data)} Sichuan samples before error")
    
    return data

def process_cantonese(cantonese_corpus, max_samples=100):
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
            audio_length = len(audio) / sampling_rate  # Calculate audio length in seconds
            gender = row.get('gender', None)  # Get gender if available
            data.append((audio, sampling_rate, label, text, audio_length, gender))
            if i >= max_samples - 1:
                break
    except Exception as e:
        print(f"Error processing Cantonese corpus: {e}")
        print(f"Processed {len(data)} Cantonese samples before error")
    
    return data

# Process all datasets
print("\n=== Processing Datasets ===")
shanghai_data = process_shanghai(shanghai_corpus)
print(f"Shanghai: {len(shanghai_data)} samples")
time.sleep(5) #api rate limit

mandarin_data = process_mandarin(mandarin_corpus)
print(f"Mandarin: {len(mandarin_data)} samples")
time.sleep(5) #api rate limit

sichuan_data = process_sichuan(sichuan_corpus)
print(f"Sichuan: {len(sichuan_data)} samples")
time.sleep(10) #longer wait time before Cantonese to avoid rate limiting

cantonese_data = process_cantonese(cantonese_corpus)
print(f"Cantonese: {len(cantonese_data)} samples")

combined_data = shanghai_data + mandarin_data + sichuan_data + cantonese_data

print(f"\n=== Final Dataset ===")
print(f"Total samples: {len(combined_data)}")

if len(combined_data) > 0:
    # Save full data with audio arrays as pickle file
    print("Saving full data with audio arrays as pickle file...")
    with open('data/loaded_data_full.pkl', 'wb') as f:
        pickle.dump(combined_data, f)
    
    # Get file size
    pickle_size = os.path.getsize('data/loaded_data_full.pkl') / (1024**2)  # MB
    print(f"Saved full data to data/loaded_data_full.pkl ({pickle_size:.2f} MB)")
    
    # Save metadata only as CSV (without audio arrays)
    print("Saving metadata as CSV file...")
    metadata_data = []
    for audio, sampling_rate, label, text, audio_length, gender in combined_data:
        metadata_data.append({
            'sampling_rate': sampling_rate,
            'label': label,
            'text': text,
            'audio_length': audio_length,
            'gender': gender,
            'audio_shape': audio.shape if hasattr(audio, 'shape') else 'No shape'
        })
    
    metadata_df = pd.DataFrame(metadata_data)
    metadata_df.to_csv('data/loaded_data.csv', index=False)
    
    csv_size = os.path.getsize('data/loaded_data.csv') / 1024  # KB
    print(f"Saved metadata to data/loaded_data.csv ({csv_size:.2f} KB)")
    
    # Show dataset summary
    print(f"\n=== Dataset Summary ===")
    print(f"Labels: {metadata_df['label'].value_counts().to_dict()}")
    print(f"Audio length range: {metadata_df['audio_length'].min():.2f}s - {metadata_df['audio_length'].max():.2f}s")
    print(f"Average audio length: {metadata_df['audio_length'].mean():.2f}s")
    
    print(f"\n=== File Summary ===")
    print(f"Full data (with audio): data/loaded_data_full.pkl ({pickle_size:.2f} MB)")
    print(f"Metadata only: data/loaded_data.csv ({csv_size:.2f} KB)")
    print(f"Use pickle file for feature extraction that requires audio arrays")
    print(f"Use CSV file for quick metadata analysis")
    
else:
    print("No data to export - all datasets failed to load")