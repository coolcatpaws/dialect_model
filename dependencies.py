# dependencies.py

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
import scipy
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from collections import Counter 
import re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer, cer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gradio as gr
print("Dependencies loaded.")
