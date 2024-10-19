import os
import pandas as pd
import torch
import google.generativeai as genai
import yaml
from transformers import AutoTokenizer, AutoModel

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Configure your Google API key
GOOGLE_API_KEY = config['google_api']['api_key']
genai.configure(api_key=GOOGLE_API_KEY)

# Set device
device = config['device'] if torch.cuda.is_available() else "cpu"

# Model configurations
model_name = config['model']['name']
embedding_model_name = config['model']['embedding_model_name']


model = genai.GenerativeModel(model_name)
# Load the tokenizer and embedding model
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)


# Load DataFrame files (example)
df = pd.read_csv(config['data']['restaurant_data_csv'])
text2_df = pd.read_csv(config['data']['restaurant_info_data_csv'], index_col=0)
