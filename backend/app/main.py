import logging
from typing import Dict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from numpy.typing import NDArray

from age_prediction_response import AgePredictionResponse
from age_predictor import AgePredictor, DeepFaceAgePredictor, CaffeAgePredictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
file_handler = logging.FileHandler('app.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI()
ROOT_URL = '/api'

predictors: Dict[str, AgePredictor] = {
    'deep_face': DeepFaceAgePredictor(),
    'caffe': CaffeAgePredictor()
}
default_predictor_name = 'deep_face'


@app.post(f'{ROOT_URL}/predict/image')
async def predict_age_from_image(image: UploadFile = File(...), predictor_name: str = None) -> AgePredictionResponse:
    logger.info('Predicting age for image ...')
    if image.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(400, detail='Invalid file type')
    try:
        image_content: bytes = await image.read()
        image_array: NDArray = np.frombuffer(image_content, np.uint8)
        img: NDArray = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return get_predictor(predictor_name).predict_age(img)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail='Error occurred while processing the image')


def get_predictor(predictor_name: str) -> AgePredictor:
    return predictors.get(predictor_name, default_predictor_name)


@app.get(f'{ROOT_URL}/health')
async def health_check() -> str:
    logger.info('Health check called')
    return 'Health check works'


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
