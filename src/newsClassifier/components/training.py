import torch
import torch.nn as NN
import torch.optim as optim
from tqdm import tqdm
import os

from logger import logging

from newsClassifier.utils.common import get_device
from newsClassifier.entity.config_entity import PrepareBaseModelConfig

logger = logging.getLogger("training_model.py")

class ModelTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader):
        self.model = model.to(get_device())
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = PrepareBaseModelConfig
        self.criterion = NN.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.device = get_device()
        self.train_losses = []
        self.validation_losses = []
        self.train_accuracies = []
        self.validation_accuracies = []

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(self.train_dataloader, desc="Training Loop"):
            review, attention_mask, label = (batch["input_ids"].to(self.device),
                                              batch["attention_mask"].to(self.device),
                                              batch["label"].to(self.device))
            self.optimizer.zero_grad()
            output = self.model(review, attention_mask)
            loss = self.criterion(output[:, 0, :], label)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            # Calculate Accuracy
            _, predicted = torch.max(output[:, 0, :], 1)
            correct_predictions += (predicted == label).sum().item()
            total_predictions += label.size(0)

        train_loss = running_loss / len(self.train_dataloader)
        train_accuracy = correct_predictions / total_predictions
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)

        return train_loss, train_accuracy

    def validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, desc="Validation Loop"):
                review, attention_mask, label = (batch["input_ids"].to(self.device),
                                                  batch["attention_mask"].to(self.device),
                                                  batch["label"].to(self.device))
                output = self.model(self.device, review, attention_mask)
                loss = self.criterion(output[:, 0, :], label)
                running_loss += loss.item()

                # Calculate Accuracy
                _, predicted = torch.max(output[:, 0, :], 1)
                correct_predictions += (predicted == label).sum().item()
                total_predictions += label.size(0)

        validation_loss = running_loss / len(self.valid_dataloader)
        validation_accuracy = correct_predictions / total_predictions
        self.validation_losses.append(validation_loss)
        self.validation_accuracies.append(validation_accuracy)

        return validation_loss, validation_accuracy

    def _save_model(self):
        
        if os.path.exists(self.config.base_model_path):
            torch.save(self.model.state_dict(), os.path.join(self.config.base_model_path, self.config.model_name))
        else:
            os.makedirs(self.config.base_model_path)
            torch.save(self.model.state_dict(), os.path.join(self.config.base_model_path, self.config.model_name))

    def train(self):
        for epoch in range(self.config.number_training_epochs):
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            validation_loss, validation_accuracy = self.validate_one_epoch()

            logger.info(f"\nEpoch {epoch + 1}/{self.config['number_training_epochs']}, "
                  f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, "
                  f"Validation Loss: {validation_loss:.3f}, Validation Accuracy: {validation_accuracy:.3f}")
            logger.info("*" * 50)
        self._save_model()
        
        

