Project Structure

```
dialect_model/
├── data/                          # All dataset files
│   ├── loaded_data.csv           # Raw metadata (6,291 samples)
│   ├── loaded_data_full.pkl      # Raw data with audio arrays (2.7GB)
│   ├── features_data.csv         # Data with extracted features
│   ├── features_data_full.pkl    # Full data with features (2.7GB)
│   ├── balanced_data.csv         # Balanced dataset metadata
│   ├── balanced_data_full.pkl    # Balanced dataset with audio
│   └── ...                       # Other balanced datasets
├── archive/                      # Other files in exploratory coding process
│   ├── data_features.ipynb       # original feature building notebook
│   ├── feature_extraction_test.ipynb      # original pulling out audio features
│   ├── replication_test.ipynb    # original testing out classification model
├── load_data.py                  # Load and process raw datasets
├── add_cantonese_data.py         # Add Cantonese data separately
├── add_features_and_balance.py   # Extract features and create balanced datasets
├── inspect_features_data.py      # Inspect data structure
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

# Environment Setup

```bash
# Create conda environment
conda create -n audio_env2 python=3.11
conda activate audio_env2

# Install dependencies
pip install -r requirements.txt

# Install brouhaha (for SNR extraction)
git clone https://github.com/marianne-m/brouhaha-vad.git
cd brouhaha-vad
pip install .
cd ..
```

# 2. Data Processing Pipeline

## Step 1: Load Raw Data
```bash
# Load Shanghai, Mandarin, Sichuan datasets
python load_data.py
```
**Output**: `data/loaded_data.csv` and `data/loaded_data_full.pkl`

## Step 2: Add Cantonese Data (Optional)
```bash
# Add Cantonese dataset (if rate limited in step 1)
python add_cantonese_data.py
```
**Output**: Updated CSV and pickle files with Cantonese data

## Step 3: Extract Features and Create Balanced Datasets
```bash
# Extract features: gender, tokens, sentiment, SNR
# Create balanced datasets
python add_features_and_balance.py
```
**Output**: 
- `data/features_data.csv` - Data with all features
- `data/features_data_full.pkl` - Full data with features
- `data/balanced_data.csv` - Balanced dataset metadata
- `data/balanced_data_full.pkl` - Balanced dataset with audio

## Step 4: Inspect Data (Optional)
```bash
# Check data structure and features
python inspect_features_data.py
```

## Dataset Information

### Raw Dataset (6,291 samples) - update once you have ec2 set up to get full 3,000 samples of each 

- **Shanghai**: 3,000 samples
- **Mandarin**: 3,000 samples  
- **Sichuan**: 91 samples
- **Cantonese**: 200 samples

### Extracted Features
- **Gender**: male/female/unknown (from audio)
- **Tokens**: Token length (from text)
- **Sentiment**: negative/neutral (from text)
- **SNR**: Signal-to-noise ratio (from audio)

### Balanced Dataset (490 samples)
- Balanced by both dialect and sentiment
- Equal representation across all combinations

## 🔧 Scripts Overview

### `load_data.py`
- Loads datasets from HuggingFace
- Handles rate limiting with delays
- Saves both CSV (metadata) and pickle (full data) formats

### `add_cantonese_data.py`
- Adds Cantonese data separately to avoid rate limits
- Updates existing CSV and pickle files

### `add_features_and_balance.py`
- Extracts all features using pre-trained models
- Creates balanced datasets for machine learning
- Saves multiple formats for different use cases

### `inspect_features_data.py`
- Inspects data structure and verifies features
- Shows dataset statistics and distributions

## Usage Examples

### Load and Process Data
```python
import pickle
import pandas as pd

# Load full data with audio arrays
with open('data/features_data_full.pkl', 'rb') as f:
    data = pickle.load(f)

# Load metadata only
df = pd.read_csv('data/features_data.csv')

# Load balanced dataset
with open('data/balanced_data_full.pkl', 'rb') as f:
    balanced_data = pickle.load(f)
```

### Data Structure
Each sample contains:
```python
(audio_array, sampling_rate, label, text, audio_length, gender, tokens, sentiment, snr)
```

## Important Notes

1. **Rate Limiting**: HuggingFace has rate limits. Scripts include delays to handle this.
2. **File Sizes**: Pickle files are large (~2.7GB) due to audio arrays.
3. **Environment**: Requires specific conda environment with audio processing libraries.
4. **Brouhaha**: Must be installed from GitHub for SNR extraction.

## Troubleshooting

### Common Issues
- **ModuleNotFoundError**: Install missing packages with `pip install -r requirements.txt`
- **Rate Limiting**: Wait and retry, or use `add_cantonese_data.py` separately
- **Memory Issues**: Use CSV files for metadata analysis, pickle for full features

### Environment Variables
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # For macOS OpenMP issues
```

## License

This project uses datasets from HuggingFace. Please check individual dataset licenses.

