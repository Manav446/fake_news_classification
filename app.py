import sys

from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
import uvicorn
from fastapi.responses import Response

from src.newsClassifier.pipeline import stage_02_model_training, stage_03_prediction
from src.newsClassifier.constants.constants import APP_HOST, APP_PORT

from src.logger import logging
from src.exception import CustomException

logger = logging.getLogger("Main_App")

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_model():
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage MODEL TRAINING started <<<<<<")
        obj = stage_02_model_training.ModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>> stage MODEL TRAINING started <<<<<<")
        return Response("Training successfull....")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail=f"Something Went wrong. {e}")

@app.post("/predict")
async def get_predictions(input_text):
    try:
        if input_text is not None:
            prediction_pipeline = stage_03_prediction.PredictionPipeline()
            response = prediction_pipeline.classify_news(input_text=input_text)
            return response
        else:
            raise HTTPException(status_code=404, detail="Input text is empty.")
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
    return

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)