import pickle
import numpy as np

def inspect_features_data():
    """Inspect the first instance of data from features_data_full.pkl"""
    
    print("Loading data/features_data_full.pkl...")
    try:
        with open('data/features_data_full.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {len(data)} samples")
    except FileNotFoundError:
        print("data/features_data_full.pkl not found!")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Get the first instance
    first_instance = data[0]
    print(f"\n=== First Instance Structure ===")
    print(f"Number of elements: {len(first_instance)}")
    
    # Expected structure: (audio, sampling_rate, label, text, audio_length, gender, tokens, sentiment, snr)
    expected_columns = ["audio", "sampling_rate", "label", "text", "audio_length", "gender", "tokens", "sentiment", "snr"]
    
    print(f"\n=== Data Breakdown ===")
    for i, (column, value) in enumerate(zip(expected_columns, first_instance)):
        print(f"{i}. {column}:")
        
        if column == "audio":
            if isinstance(value, np.ndarray):
                print(f"   Type: numpy.ndarray")
                print(f"   Shape: {value.shape}")
                print(f"   Data type: {value.dtype}")
                print(f"   Sample values: {value[:5]}...")
                print(f"   Min/Max: {value.min():.4f} / {value.max():.4f}")
            else:
                print(f"   Type: {type(value)}")
                print(f"   Value: {value}")
        
        elif column == "text":
            print(f"   Type: {type(value)}")
            print(f"   Length: {len(value)} characters")
            print(f"   Content: '{value[:100]}{'...' if len(value) > 100 else ''}'")
        
        elif column in ["sampling_rate", "audio_length", "tokens", "snr"]:
            print(f"   Type: {type(value)}")
            print(f"   Value: {value}")
        
        else:
            print(f"   Type: {type(value)}")
            print(f"   Value: {value}")
        
        print()
    
    # Check if all expected features are present
    print("=== Feature Verification ===")
    missing_features = []
    for column in expected_columns:
        if column not in [expected_columns[i] for i in range(len(first_instance))]:
            missing_features.append(column)
    
    if missing_features:
        print(f"Missing features: {missing_features}")
    else:
        print("All expected features are present!")
    
    # Show some statistics
    print(f"\n=== Dataset Statistics ===")
    labels = [row[2] for row in data]
    sentiments = [row[7] for row in data]
    genders = [row[5] for row in data]
    snrs = [row[8] for row in data]
    token_lengths = [row[6] for row in data]
    
    print(f"Labels: {set(labels)}")
    print(f"Sentiments: {set(sentiments)}")
    print(f"Genders: {set(genders)}")
    print(f"SNR range: {min(snrs):.2f} - {max(snrs):.2f} dB")
    print(f"Token length range: {min(token_lengths)} - {max(token_lengths)}")
    
    # Show distribution
    from collections import Counter
    print(f"\nLabel distribution: {Counter(labels)}")
    print(f"Sentiment distribution: {Counter(sentiments)}")
    print(f"Gender distribution: {Counter(genders)}")

if __name__ == "__main__":
    inspect_features_data()
