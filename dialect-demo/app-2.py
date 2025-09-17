import torch
import torch.nn as nn
import gradio as gr
import librosa
import numpy as np
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ----------------- Model definition -----------------
class LanNetBinary(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.linear2 = nn.Linear(hidden_dim, 192)
        self.linear3 = nn.Linear(192, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        x = self.linear2(last)
        x = self.linear3(x)
        return x

# ----------------- Load config + model -----------------
REPO_ID = "karenlu653/dialect_model_naive"

# Load configs
config = json.load(open(hf_hub_download(REPO_ID, "config.json"), "r"))
preproc = json.load(open(hf_hub_download(REPO_ID, "preprocessor_config.json"), "r"))
label_map = json.load(open(hf_hub_download(REPO_ID, "label_mapping.json"), "r"))

# Instantiate model with correct params
model = LanNetBinary(
    input_dim=config.get("input_dim", 40),
    hidden_dim=config.get("hidden_dim", 512),
    num_layers=config.get("num_layers", 2)
)
state_dict = load_file(hf_hub_download(REPO_ID, "model.safetensors"))
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------- Feature extraction -----------------
def extract_features(y, sr):
    n_mels = preproc.get("n_mels", 40)
    n_fft = preproc.get("n_fft", 400)
    hop_length = preproc.get("hop_length", 160)
    max_len = preproc.get("max_len_frames", 200)

    # Resample if needed
    target_sr = preproc.get("sampling_rate", 16000)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    fbanks = librosa.power_to_db(mel).T

    # Pad/truncate
    if fbanks.shape[0] < max_len:
        fbanks = np.pad(fbanks, ((0, max_len - fbanks.shape[0]), (0, 0)), mode="constant")
    else:
        fbanks = fbanks[:max_len, :]

    return torch.tensor(fbanks, dtype=torch.float32).unsqueeze(0)  # (1, T, F)

# ----------------- Prediction function -----------------

import tempfile, shutil

def predict(audio_path):
    if not audio_path:
        return "No audio provided"

    # Copy to a safe temp file
    tmp_path = tempfile.mktemp(suffix=".wav")
    shutil.copy(audio_path, tmp_path)

    import soundfile as sf
    y, sr = sf.read(tmp_path, dtype="float32")
    if len(y) == 0:
        return "No audio detected, please try again."

    feats = extract_features(y, sr).to(device)
    with torch.no_grad():
        logits = model(feats)
        pred = int(logits.argmax(dim=1))

    return label_map.get(str(pred), str(pred))

def evaluate_dataset(repo_id):
    try:
        ds = load_dataset(repo_id, split="train")
    except Exception as e:
        return f"Could not load dataset: {e}"

    y_true, y_pred = [], []
    for row in ds:
        y = np.array(row["audio"], dtype=np.float32)
        sr = preproc.get("sampling_rate", 16000)
        feats = extract_features(y, sr).to(device)
        with torch.no_grad():
            logits = model(feats)
            pred = int(logits.argmax(dim=1))
        y_pred.append(pred)
        y_true.append(row["label"])

    # Confusion matrix
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[label_map[str(l)] for l in labels],
        yticklabels=[label_map[str(l)] for l in labels],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix\n{repo_id}")
    plt.tight_layout()

    return fig

# ----------------- Gradio Interface -----------------

with gr.Blocks() as demo:
    with gr.Tab("Single Prediction"):
        gr.Interface(
            fn=predict,
            inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
            outputs="text",
            description = "Upload or record audio to classify if this is the Shanghai dialect!", 
            live=False
        )
    with gr.Tab("Dataset Evaluation"):
        dataset_input = gr.Textbox(
            label="Hugging Face Dataset ID",
            value="karenlu653/dialect_model_demo",  # default
            placeholder="e.g. username/repo_name"
        )
        eval_btn = gr.Button("Run Evaluation")
        eval_output = gr.Plot()
        eval_btn.click(evaluate_dataset, inputs=dataset_input, outputs=eval_output)

if __name__ == "__main__":
    demo.launch()