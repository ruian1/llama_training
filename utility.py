import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score


def group_roc_auc_score(y_true, y_pred, test_group):
    # auc = roc_auc_score(np.array(y_true), np.array(y_pred))

    tmp_end = 0
    result_weight = 0.0
    total_weight = 0.0
    for g in test_group:
        tmp_start = tmp_end
        tmp_end = tmp_start + g
        tmp_true = y_true[tmp_start: tmp_end]
        tmp_pred = y_pred[tmp_start: tmp_end]
        label_len = len(list(set(tmp_true)))
        if label_len == 1:
            continue
        tmp_auc = roc_auc_score(np.array(tmp_true), np.array(tmp_pred))
        tmp_len = g
        result_weight += tmp_auc * tmp_len
        total_weight += tmp_len
    gauc = result_weight
    
    return gauc

def compute_metrics(p):
    
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    
    probas = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    positive_class_probs = probas[:, 1]
    
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions, average="weighted")
    precision = precision_score(y_true=labels, y_pred=predictions, average="weighted")
    auc = roc_auc_score(y_true=labels, y_score=positive_class_probs)
    
    return {"accuracy": accuracy, "recall": recall, "precision": precision, "auc": auc}
