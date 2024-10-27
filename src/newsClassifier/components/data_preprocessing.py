import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from newsClassifier.entity.config_entity import TrainingDataPreprationConfig, PrepareBaseModelConfig


class FakeNewClassificationDataset(Dataset):
  def __init__(self, data, label, tokenizer, max_length):
    self.data = data
    self.label = label
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    encoding = self.tokenizer(
        self.data.iloc[index],
        max_length=self.max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "label": torch.tensor(self.label.iloc[index], dtype=torch.long)
    }

class FakeNewsClassifierDataLoader:
    def __init__(self):
        self.csv_path = TrainingDataPreprationConfig.input_data_path
        self.max_length = PrepareBaseModelConfig.max_sequence_length
        self.batch_size = PrepareBaseModelConfig.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(TrainingDataPreprationConfig.tokenizer_model_id)
        self.data = self.load_data()
        self.train_loader, self.valid_loader, self.test_loader = self.prepare_data()
        

    def load_data(self):
        self.data = pd.read_csv(self.csv_path)
        self.data["label"] = self.data["label"].map({"fake": 0, "true": 1})
        return self.data.sample(frac=1).reset_index(drop=True)

    def prepare_data(self):
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data["text"], self.data["label"], test_size=TrainingDataPreprationConfig.train_test_split, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=TrainingDataPreprationConfig.train_test_split, random_state=42
        )

        train_dataset = FakeNewClassificationDataset(X_train, y_train, self.tokenizer, self.max_length)
        valid_dataset = FakeNewClassificationDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = FakeNewClassificationDataset(X_test, y_test, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.save_tokenizer(TrainingDataPreprationConfig.tokenizer_path)

        return train_loader, valid_loader, test_loader
    
    def save_tokenizer(self, tokenizer_path: str):
       try:
        if os.path.exists(tokenizer_path):
            self.tokenizer.save_pretrained(tokenizer_path)
        else:
            os.makedirs(tokenizer_path)
            self.tokenizer.save_pretrained(tokenizer_path)
       except Exception as e:
          print(e)
        
    