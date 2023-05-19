import os
import json

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tabulate import tabulate

def reverse_sigmoid(arr):
    arr = arr.numpy()
    return np.log(arr / (1 - arr))

def softmax(arr):
    # print(arr)
    return np.exp(arr) / np.sum(np.exp(arr))

def gauc(input_file_name):
    FOLDER_PATH = os.getcwd()
    input_file_path = os.path.join(FOLDER_PATH, input_file_name)

    with open(input_file_path, 'rb') as file:
        df = pickle.load(file)

    df['predicted_logits'] = df['predicted_label'].apply(lambda x: reverse_sigmoid(x))
    df['predicted_softmax'] = df['predicted_logits'].apply(lambda x: softmax(x[0]))
    
    df['predicted_ones'] = df['predicted_softmax'].apply(lambda x: x[1])
    df['predicted_zeros'] = df['predicted_softmax'].apply(lambda x: x[0])


    # Convert probabilities into predicted labels
    df['predicted_label'] = df.apply(lambda row: 1 if row['predicted_ones'] > row['predicted_zeros'] else 0, axis=1)

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(df['label'], df['predicted_label'])
    precision = precision_score(df['label'], df['predicted_label'])
    recall = recall_score(df['label'], df['predicted_label'])
    
    
    groups = df.groupby("task_id")

    gauc = 0.0
    for _, entry in groups:
        tmp_auc = roc_auc_score(entry['label'], entry['predicted_ones'])
        weight = float(len(entry))
    
        gauc += weight * tmp_auc

    gauc /= float(len(df))
    
    print(input_file_path)
    
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "GAUC": gauc}
    
    return metrics

input_file_name = "results/aisourcing_20220102_20220103_output_checkpoint-2349.pkl"
print(tabulate(gauc(input_file_name).items(), headers=["Metric", "Value"], tablefmt="pretty"))
input_file_name = "results/aisourcing_20220102_20220103_output_checkpoint-4699.pkl"
print(tabulate(gauc(input_file_name).items(), headers=["Metric", "Value"], tablefmt="pretty"))
input_file_name = "results/aisourcing_20220102_20220103_output_checkpoint-7049.pkl"
print(tabulate(gauc(input_file_name).items(), headers=["Metric", "Value"], tablefmt="pretty"))
input_file_name = "results/aisourcing_20220102_20220103_output_checkpoint-9396.pkl"
print(tabulate(gauc(input_file_name).items(), headers=["Metric", "Value"], tablefmt="pretty"))