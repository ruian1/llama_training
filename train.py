import json
import os
import pickle
from time import time

import pandas as pd
import torch
import torch.nn as nn
import wandb

from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (AdamW, LlamaTokenizer, Trainer, TrainingArguments,
                          LlamaForSequenceClassification)

from llama_dataloader import BinaryClassificationDataset
# from llama_model_debug import LlamaForSequenceClassification
from llama_save_strategy import SaveTokenizerCallback
from llama_peft_callback import SavePeftModelCallback
from utility import compute_metrics


with open('config/config_peft.json', 'r') as f:
    config = json.load(f)

print(config)

batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
max_sql_len = config['model']['max_sql_len']

load_in_8bit = config['training']['load_in_8bit']
fp16 = config['training']['fp16']
gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
learning_rate = config['training']['learning_rate']

save_strategy = config['training']['save_strategy']
max_steps = config['training']['max_steps']
save_steps = config['training']['save_steps']
# warmup_steps = config['training']['warmup_steps']

training_data_file = config['data']['training_data_file']

wandb_key = config['wandb']['api_key']

wandb.login(key=wandb_key)

from datetime import datetime

# Get the current date and time
current_time = datetime.now()

# Format the current time as a string
time_string = current_time.strftime("%d-%m-%Y-%H-%M")

wandb.init(project="llama_test", 
           name=time_string, 
           config=config,)


os.environ["WANDB_MODE"] = "dryrun"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Training with {device}")

FOLDER_PATH = os.getcwd()

model_path = os.path.join(FOLDER_PATH, "weights/llama-7b")

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

    # model = model.half()
    model = model.to(device)

print("before training")
# print(model.state_dict()['base_model.model.score.weight'])
print(model.state_dict()['base_model.model.score.modules_to_save.default.weight'])
print("FOLDER_PATH: ", FOLDER_PATH)

directory = os.path.join(FOLDER_PATH, "data")
training_dir = os.path.join(FOLDER_PATH, "training")

# #train_data_01_4_200.csv
# def prepare_data():
#     train_df = pd.read_csv(os.path.join(directory, training_data_file), delimiter='$')
#     train_df['jd_reusme'] = train_df['query'] + "\n" + train_df['resume']
#     train_df = train_df.dropna(axis=0)
#     nan_age_df = train_df[train_df['jd_reusme'].isna()]
#     train_data, valid_data = train_test_split(train_df, test_size=0.1, random_state=42)
#     return train_data, valid_data

def prepare_data():
    # train_data_01_4_200.csv
    train_file = os.path.join(directory, training_data_file)
    with open(train_file, 'r') as f:    
        entries = [json.loads(line) for line in f.readlines()]

    train_df = pd.DataFrame(entries, columns=entries[0].keys())
    train_df['jd_reusme'] = train_df['astask'] + "\n" + train_df['profile']
    train_df = train_df.dropna(axis=0)
    train_data, valid_data = train_test_split(train_df, test_size=0.1, random_state=42)
    # train_texts, train_labels = train_data['jd_reusme'].tolist(), train_data['label'].tolist()
    # valid_texts, valid_labels = valid_data['jd_reusme'].tolist(), valid_data['label'].tolist()
    return train_data, valid_data

train_data, valid_data = prepare_data()

tokenizer.add_special_tokens({"pad_token": "<pad>"})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = 32000

# if tokenizer.pad_token is None:
#     print("adding pad token....")
#     print("tokenizer size before: ", tokenizer.vocab_size)
#     tokenizer.add_special_tokens({"pad_token": "<pad>"})
#     model.resize_token_embeddings(len(tokenizer))
#     print("tokenizer size after: ", tokenizer.vocab_size)

    
train_dataset = BinaryClassificationDataset(train_data, tokenizer, max_length=max_sql_len)
valid_dataset = BinaryClassificationDataset(valid_data, tokenizer, max_length=max_sql_len)
    
# num_training_steps = epochs * (len(train_dataset) // batch_size)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

output_directory = os.path.join(training_dir, f'./results/{time_string}')

print("#"*100)
print("epochs: ", epochs)
# print("num_training_steps: ", num_training_steps)

num_training_steps = len(train_dataset) // batch_size  // gradient_accumulation_steps 
warmup_steps = int(num_training_steps * 0.1)

print("training steps: ", num_training_steps)
print("warmup steps: ", warmup_steps)

training_args = TrainingArguments(
    output_dir=output_directory,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps, 
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir=os.path.join(training_dir, './logs'),
    logging_steps=10,
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=5,
    fp16=fp16,
    tf32=True,
    save_strategy=save_strategy,
    # max_steps=max_steps,
    # save_steps=save_steps,
    report_to="wandb",
    # gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[SaveTokenizerCallback(tokenizer, os.path.join(output_directory, "tokenizer")), SavePeftModelCallback],
    # compute_metrics=compute_metrics,
)

trainer.train()

print("Trainig done. Saving model and tokenizer")

model.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)
print("tokenizer size ", tokenizer.vocab_size)

print("after training")
# print(model.state_dict()['base_model.model.score.weight'])
print(model.state_dict()['base_model.model.score.modules_to_save.default.weight'])
print("Trainig done. Saving model and tokenizer to disk at: ", output_directory)
