import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

def compute_metrics(p):
    print("in compute_metrics!!!")
    print(p)
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions, average="weighted")
    precision = precision_score(y_true=labels, y_pred=predictions, average="weighted")
    return {"accuracy": accuracy, "recall": recall, "precision": precision}
