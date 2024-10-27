from newsClassifier.components.data_ingestion import DataIngestion
from newsClassifier.entity.config_entity import DataIngestionConfig
from logger import logging

STAGE_NAME = "Data Ingestion stage"

logger = logging.getLogger("DataIngestionPipeline")

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        data_ingestion = DataIngestion(config=DataIngestionConfig)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e