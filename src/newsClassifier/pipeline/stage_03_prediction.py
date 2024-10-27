import os
import sys
import torch
from transformers import AutoTokenizer
from newsClassifier.entity.config_entity import TrainingDataPreprationConfig, PrepareBaseModelConfig
from newsClassifier.components import prepare_base_model
from newsClassifier.utils.common import get_device

from logger import logging
from exception import CustomException

logger = logging.getLogger("Prediction_Pipeline")

class PredictionPipeline:
    def __init__(self):
        pass

    @staticmethod
    def _load_model():
        logger.info("Loading pretrained model from directory........")
        try:
            base_model = prepare_base_model.PrepareBaseModel()
            model = base_model.get_base_model()
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        PrepareBaseModelConfig.base_model_path,
                        PrepareBaseModelConfig.model_name
                    ),
                    map_location=get_device(),
                )
            )
        except Exception as e:            
            logger.error(e)
            raise CustomException(e, sys)
        return model
    
    @staticmethod
    def _get_data_tokenized(input_text:str):
        logger.info("Loading pretained tokenizer.........")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                TrainingDataPreprationConfig.tokenizer_path
            )
            input_batch = tokenizer(
                input_text,
                max_length=PrepareBaseModelConfig.max_sequence_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                return_tensors="pt"
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        return {
            "input_ids": input_batch["input_ids"],
            "attention_mask": input_batch["attention_mask"]
        }

    def classify_news(self, input_text: str):
        logger.info("going to classify the input news........")
        try:
            model = PredictionPipeline._load_model()
            model = model.to(get_device())
            tokenized_text = PredictionPipeline._get_data_tokenized(input_text=input_text)
            model.eval()
            with torch.no_grad():
                input, attention_mask = tokenized_text["input_ids"].to(get_device()), tokenized_text["attention_mask"].to(get_device())
                output = model(input, attention_mask)
                _, prediction = torch.max(output[:,0,:], 1)
            
            output = "True" if prediction[0] == 1 else "Fake"
            logger.info(f"Prediction Label: {prediction[0]} and Final output: {output}")

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        return output
