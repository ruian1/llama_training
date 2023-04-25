import json
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import LlamaTokenizer

from llama_model_debug import LlamaForSequenceClassification

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
checkpoint = inference_config['checkpoint']

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



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_tokenizer(training_dir, checkpoint)
    model.to(device)

    text = "quality analyst or quality assurance analyst or quality assurance engineer or quality analyst ii or test analyst or test analyst i or test analyst ii or test analyst iii or test analyst iv or sr qa analyst or senior quality analyst or senior quality assurance analyst or sr test analyst or sr test engineer or qa specialist or qa lead or quality assurance lead or qa test lead or test lead, with 0-2 or 2-4 or 4-6 or 8-10 years of experience. familiar with skills: functional testing. in the areas of united states or canada.$title: test analyst, sdk, game, service^years: 0-2^location: irvine, california, united states^company size: 5001-10000^work experience: ^1. senior test analyst i - b&op sdk & game services, blizzard entertainment, 2023~2100^2. senior test analyst i - b&op player engagements, blizzard entertainment, 2021~2023^3. senior test analyst - itcapps, blizzard entertainment, 2020~2021^degree: master^major: software engineering, software testing and verification^school: edx, university of maryland^skills: integration testing, exploratory testing, bdd, css, rest apis, testrail, confluence, api testing, java, api automation, html5, cascading, rest api, git, application, javascript, open api, smoke testing, test case, mysql, game, ui automation, api, software quality assurance, test, microservices, qa automation, selenium, development, estimation, regression testing, e2e, json, webdriver, automation, postman api, swagger api, sdk, service, framework, agile methodology, jira, cucumber, ui, regression, integration, jmeter, selenium webdriver, quality assurance, web testing, qa testing, testing, agile methodologies^industry: entertainment, computer games^has email: True^has phone number: True"
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_sql_len, return_tensors="pt")
    # inputs = {key: tensor[0] for key, tensor in inputs.items()}
    # print(inputs.keys())
    print("inputs are", inputs)

    with torch.no_grad():
        if model.training:
            print("Model is in train mode")
        else:
            print("Model is in eval mode")
            
        logits = model(**inputs).logits
        print("logits are", logits)
        predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
        print("predicted_class_ids are", predicted_class_ids)
