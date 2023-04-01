import os
import pickle
from time import time

import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (AdamW,
                          LlamaTokenizer, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)

from rui_model_debug import LlamaForSequenceClassification

from llama_dataloader import BinaryClassificationDataset

os.environ["WANDB_MODE"] = "dryrun"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

FOLDER_PATH = os.getcwd()

model_path = os.path.join(FOLDER_PATH, "weights/llama-7b")

load_in_8bit = True

tokenizer = LlamaTokenizer.from_pretrained(model_path)
if load_in_8bit:
    model = LlamaForSequenceClassification.from_pretrained(model_path, 
                                                           load_in_8bit=load_in_8bit, 
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")

if not load_in_8bit:
    model = LlamaForSequenceClassification.from_pretrained(model_path)
    print("moving model to PEFT")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1, 
        # target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model = model.half()
    model = model.to(device)

print("FOLDER_PATH: ", FOLDER_PATH)

directory = os.path.join(FOLDER_PATH, "data")
training_dir = os.path.join(FOLDER_PATH, "training")
train_df = pd.read_csv(os.path.join(directory, 'train_data_01_4_200.csv'), delimiter='$')

train_df['jd_reusme'] = train_df['query'] + "\n" + train_df['resume']
train_df = train_df.dropna(axis=0)
nan_age_df = train_df[train_df['jd_reusme'].isna()]
nan_age_df

train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
train_texts, train_labels = train_data['jd_reusme'].tolist(), train_data['label'].tolist()
valid_texts, valid_labels = valid_data['jd_reusme'].tolist(), valid_data['label'].tolist()


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    
    
MAX_LENGTH = 512
train_dataset = BinaryClassificationDataset(train_data, tokenizer, max_length=MAX_LENGTH)
valid_dataset = BinaryClassificationDataset(valid_data, tokenizer, max_length=MAX_LENGTH)
    
epochs = 1
batch_size = 1
lr = 2e-5
warmup_steps = 500
num_training_steps = epochs * (len(train_dataset) // batch_size)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

fp16 = False if load_in_8bit else True

training_args = TrainingArguments(
    output_dir=os.path.join(training_dir, './results'),
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=8,
    gradient_accumulation_steps=4, 
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir=os.path.join(training_dir, './logs'),
    logging_steps=10,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    fp16=fp16,
    tf32=True,
    max_steps=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=valid_dataset,
)

trainer.train()