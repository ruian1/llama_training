import json
import sys
import os

import argparse
import torch
from time import time

from peft import PeftConfig, PeftModel
from transformers import LlamaTokenizer

from llama_model_debug import LlamaForSequenceClassification

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='Run the model with a specific checkpoint.')

# Add the checkpoint argument
parser.add_argument('checkpoint', type=str, help='The checkpoint to load.')

# Parse the arguments
args = parser.parse_args()
checkpoint = args.checkpoint

# get the max sql length from the training config file
with open('config/config_peft.json', 'r') as f:
    train_config = json.load(f)
    print(train_config)
max_sql_len = train_config['model']['max_sql_len']

# get inference path for model and tokenizer
with open('config/config_inference.json', 'r') as f:
    inference_config = json.load(f)
    print(inference_config)
training_dir = inference_config['training_dir']
# checkpoint = inference_config['checkpoint']

print(f"checkpoint is {checkpoint}")

def load_model_tokenizer(training_dir, checkpoint):
    FOLDER_PATH = os.getcwd()

    tokenizer_directory = os.path.join(FOLDER_PATH, f"training/results/{training_dir}/tokenizer")
    print("tokenizer_directory is", tokenizer_directory)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_directory)
    print("tokenizer size is", tokenizer.vocab_size)

    peft_directory = os.path.join(FOLDER_PATH, f"training/results/{training_dir}/{checkpoint}/adapter_model")
    print("peft_directory is", peft_directory)
    peft_config = PeftConfig.from_pretrained(peft_directory)
    print("peft_config is", peft_config)

    print(f"Loading model from base model at {peft_config.base_model_name_or_path}")
    model = LlamaForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
    print("1"* 100)
    print(model.state_dict()['score.weight'])

    print(f"Loading model from PEFT at {peft_directory}")
# peft_model_id = os.path.join(FOLDER_PATH, "training/results")
    model = PeftModel.from_pretrained(model, model_id=peft_directory)
    model.eval()

    print("2"* 100)
    print(model.state_dict()['base_model.model.score.modules_to_save.default.weight'])

    print("tokenizer size is", tokenizer.vocab_size)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = 32000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer,model



def inference(max_sql_len, device, tokenizer, model, text, verbose=False):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_sql_len, return_tensors="pt")
    # inputs = {key: tensor[0] for key, tensor in inputs.items()}
    # print(inputs.keys())
    if verbose: 
        print("inputs are", inputs)
    inputs = inputs.to(device)
    with torch.no_grad():
        if verbose:
            if model.training:
                print("Model is in train mode")
            else:
                print("Model is in eval mode")
        t0 = time()
        logits = model(**inputs).logits
        if verbose: 
            print("logits are", logits)
            print("torch.sigmoid(logits)", torch.sigmoid(logits))
        predicted_class_ids = torch.arange(0, logits.shape[-1], device=device)[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
        if verbose: 
            print("predicted_class_ids are", predicted_class_ids)
        t1 = time()
        if verbose: 
            print(f"inference costs {t1-t0:.2f}")

    return torch.sigmoid(logits).cpu()

def read_data(file_path):
    data_list = []

    f_r = open(file_path, mode="r", encoding="utf-8")
    line = f_r.readline().strip()

    while line:
        line_json = json.loads(line)
        task_id = line_json["task_id"]
        # profile_id = line_json["profile_id"]
        astask = line_json["astask"]
        profile = line_json["profile"]
        label = line_json["label"]

        tmp_data = [task_id, astask, profile,  label]
        data_list.append(tmp_data)

        line = f_r.readline().strip()

    f_r.close()

    return data_list

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_tokenizer(training_dir, checkpoint)
    print("device is", device)
    model.to(device)

    # astask = "titles: ['marketing coordinator', 'marketing specialist', 'marketing manager', 'email marketing', 'webinar marketing'], exp_years: ['2-4', '4-6', '6-8'], location: ['california state', 'washington state', 'oregon state', 'arizona state', 'nevada state', 'massachusetts state', 'new york state', 'texas state', 'florida state', 'georgia state', 'north carolina state', 'south carolina state', 'maryland state', 'pennsylvania state'], "
    # profile = "title: ['channel marketing manager', 'head of marketing']\nyears: 2-4\nlocation: san francisco bay area, california, united states\ncompany size: 51-200\nwork experience: \n1. channel marketing manager, b-stock solutions, 2022~2100\n2. head of marketing, d\u00e9rive marketing & public relations llc., 2021~2100\n3. marketing associate, b-stock solutions, 2021~2022\ndegree: bachelor\nmajor: ['marketing and global business', 'international business']\nschool: [\"saint mary's college of california\", 'john cabot university', 'mercy high school burlingame']\nskills: ['marketing plan', 'content strategy', 'corporate', 'gotowebinar', 'analytics', 'growth marketing', 'product marketing', 'management', 'business', 'demand generation', 'creativity skills', 'email marketing', 'communication strategy', 'marketing campaigns', 'social media', 'social', 'research', 'segmentation', 'marketing automation', 'hubspot', 'market research', 'production', 'advertising', 'enterprise marketing', 'direct mail marketing', 'field marketing', 'consumer insight', 'customer service', 'b2c', 'multi-channel marketing', 'microsoft office', 'public relation', 'microsoft powerpoint', 'sales', 'administration', 'google suite', 'account', 'public relations', 'website', 'account management', 'b2b marketing', 'kpi reports', 'content creation', 'strategy', 'conversion rate', 'channel marketing', 'roi', 'direct marketing', 'communication', 'consumer', 'social media marketing', 'teamwork', 'marketing strategy', 'mass email marketing', 'marketing campaign', 'salesforce', 'campaign', 'digital marketing', 'creative development', 'google drive', 'customer relationship management crm', 'crm databases', 'direct mail fundraising', 'small business marketing', 'online marketing', 'microsoft word', 'mail', 'facebook marketing', 'team leadership', 'leadership', 'influencer marketing', 'editing', 'time management', 'company newsletters', 'collateral', 'google analytics', 'microsoft excel', 'marketing', 'local marketing']\nindustry: ['marketing and advertising', 'logistics and supply chain']\nhas email: True\nhas phone number: True\n"
    # text = astask + "\n" + profile

    # inference(max_sql_len, device, tokenizer, model, text, verbose=True)

    import pandas as pd
    from tqdm import tqdm
    
    tqdm.pandas()
    
    FOLDER_PATH = os.getcwd()
    input_file_name = "data/aisourcing_20220102_20220103.json"
    input_file_path = os.path.join(FOLDER_PATH, input_file_name)
    
    curr_data_list = read_data(input_file_path)
    df = pd.DataFrame(curr_data_list, columns=["task_id", "astask", "profile", "label"])
    df['jd_reusme'] = df['astask'] + "\n" + df['profile']
    
    # df['predicted_label'] = df.apply(lambda x: inference(max_sql_len, device, tokenizer, model, 
    #                                                      x["jd_reusme"]), axis=1)
    
    df['predicted_label'] = df.progress_apply(lambda x: inference(max_sql_len, device, 
                                                                  tokenizer, model, 
                                                                  x["jd_reusme"]), axis=1)
    
    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        df.to_json(f"results/aisourcing_20220102_20220103_output_{checkpoint}.json", orient="records")
    except Exception as e:
        print("Error when saving json file", e)
    
    try:
        import pickle
        with open(f"results/aisourcing_20220102_20220103_output_{checkpoint}.pkl", "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        print("Error when saving pickle file", e)