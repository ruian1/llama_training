import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

class BinaryClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]
    #     text, label = row["jd_reusme"], row["label"]
    #     inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
    #     inputs["labels"] = torch.tensor(label, dtype=torch.long)
    #     return inputs
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text, label = row["jd_reusme"], row["label"]
        print("text is", text)
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        print("inputs are", inputs)
        inputs = {key: tensor[0] for key, tensor in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs