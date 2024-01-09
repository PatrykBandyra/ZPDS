import logging
from io import BytesIO
from typing import Union

import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from segmentation import perform_selfie_segmentation_with_bg_swap, perform_selfie_segmentation_white_bg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
file_handler = logging.FileHandler('app.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI()
ROOT_URL = '/api'


def create_response_image(img: np.ndarray) -> StreamingResponse:
    pillow_img = Image.fromarray(img)
    response_img = BytesIO()
    pillow_img.save(response_img, 'JPEG')
    response_img.seek(0)

    return StreamingResponse(response_img, media_type='image/jpeg')


@app.post(f'{ROOT_URL}/segmentation')
async def image_segmentation(image: UploadFile = File(...),
                             bg_image: Union[UploadFile, None] = None,
                             bg_color: str = None) -> StreamingResponse:
    logger.info('/segmentation endpoint called')
    if image.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(400, detail='Invalid file type')
    if bg_image is not None:
        result_img = perform_selfie_segmentation_with_bg_swap(image.file, bg_image.file)
    else:
        result_img = perform_selfie_segmentation_white_bg(image.file, bg_color)
    return create_response_image(result_img)


@app.get(f'{ROOT_URL}/health')
async def health_check() -> str:
    logger.info('/health endpoint called')
    return '[SGM]: Health check works'


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
