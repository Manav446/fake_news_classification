import torch

from newsClassifier.components import prepare_base_model, data_preprocessing, training

from logger import logging

logger = logging.getLogger("BaseModelCreationPipeline")
STAGE_NAME = "Prepare Base Model stage"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f"Device using: {device}")
        
        logger.info("Going to load Model........")
        base_model = prepare_base_model.PrepareBaseModel()
        model = base_model.get_base_model()

        logger.info("Going to load dataloaders for training..........")
        data_loader = data_preprocessing.FakeNewsClassifierDataLoader()
        data_loader.load_data()
        train_loader, valid_loader, test_loader = data_loader.prepare_data()

        logger.info("Training Model starts................")        
        trainer = training.ModelTrainer(model = model, train_dataloader=train_loader, valid_dataloader=valid_loader)
        trainer.train()


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
