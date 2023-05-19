import json
import os

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from tqdm import tqdm

from inference import load_model_tokenizer
from llama_dataloader import BinaryClassificationDataset
from utility import compute_metrics

parser = argparse.ArgumentParser(description='Run the model with a specific checkpoint.')

# Add the checkpoint argument
parser.add_argument('checkpoint', type=int, help='The checkpoint to load.')

# Parse the arguments
args = parser.parse_args()
checkpoint = args.checkpoint

FOLDER_PATH = os.getcwd()
directory = os.path.join(FOLDER_PATH, "data")
training_dir = os.path.join(FOLDER_PATH, "training")
device = "cuda" if torch.cuda.is_available() else "cpu"

with open('config/config_peft.json', 'r') as f:
    config = json.load(f)
valid_data_file = config['data']['valid_data_file']
max_sql_len = config['model']['max_sql_len']

def prepare_data(training_data_file):
    print("Loading file: ", training_data_file)

    train_file = os.path.join(directory, training_data_file)
    with open(train_file, 'r') as f:    
        entries = [json.loads(line) for line in f.readlines()]

    train_df = pd.DataFrame(entries, columns=entries[0].keys())
    train_df['jd_reusme'] = train_df['astask'] + "\n" + train_df['profile']
    train_df = train_df.dropna(axis=0)
    # train_data, valid_data = train_test_split(train_df, test_size=0.1, random_state=42)
    # train_texts, train_labels = train_data['jd_reusme'].tolist(), train_data['label'].tolist()
    # valid_texts, valid_labels = valid_data['jd_reusme'].tolist(), valid_data['label'].tolist()
    print("Number of training samples: ", len(train_df))


model_path = os.path.join(FOLDER_PATH, "weights/llama-7b")
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"pad_token": "<pad>"})

valid_data = prepare_data(valid_data_file)
valid_dataset = BinaryClassificationDataset(valid_data, tokenizer, max_length=max_sql_len)
eval_dataloader = DataLoader(valid_dataset, batch_size=6)  # Set a smaller batch size

with open('config/config_inference.json', 'r') as f:
    inference_config = json.load(f)
    print(inference_config)

training_dir = inference_config['training_dir']
# checkpoint = inference_config['checkpoint']

tokenizer, model = load_model_tokenizer(training_dir, checkpoint)
model = model.to(device)
# for batch in eval_dataloader:
#     print(batch)
#     print(batch.keys())
    
#     break

print("model.device:", model.device)

model.eval()

all_logits = []
all_labels = []

# for batch in eval_dataloader:
for batch in tqdm(eval_dataloader, desc="Evaluating"):

    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

    all_logits.append(logits.detach().cpu().numpy())
    all_labels.append(labels.detach().cpu().numpy())

all_logits = np.concatenate(all_logits, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

eval_metrics = compute_metrics((all_logits, all_labels))
print(eval_metrics)
