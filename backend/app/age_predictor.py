from abc import ABC, abstractmethod
from typing import List, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace
from numpy.typing import NDArray

from age_prediction_response import AgePredictionResponse


class AgePredictor(ABC):

    @abstractmethod
    def predict_age(self, image: NDArray) -> AgePredictionResponse:
        raise NotImplementedError()


class DeepFaceAgePredictor(AgePredictor):

    def predict_age(self, image: NDArray) -> AgePredictionResponse:
        results = DeepFace.analyze(image, actions=['age'])
        return DeepFaceAgePredictor.map_to_response(results)

    @staticmethod
    def map_to_response(results: List[Dict[str, Any]]) -> AgePredictionResponse:
        predictions = []
        for result in results:
            region = AgePredictionResponse.AgePrediction.Region(**result['region'])
            age_prediction = AgePredictionResponse.AgePrediction(result['age'], region, result['face_confidence'])
            predictions.append(age_prediction)
        return AgePredictionResponse(predictions)


class CaffeAgePredictor(AgePredictor):

    def __init__(self):
        self.detector = cv2.CascadeClassifier('app/haarcascade_frontalface_default.xml')
        self.age_model = cv2.dnn.readNetFromCaffe('app/age.prototxt', 'app/dex_chalearn_iccv2015.caffemodel')
        self.age_indexes = np.array([i for i in range(0, 101)])

    def predict_age(self, image: NDArray) -> AgePredictionResponse:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detectMultiScale(img_rgb, 1.3, 5)

        predictions = []
        for face in faces:
            x, y, w, h = face
            detected_face = img_rgb[int(y):int(y + h), int(x):int(x + w)]
            detected_face_resized = cv2.resize(detected_face, (224, 224))
            detected_face_blob = cv2.dnn.blobFromImage(detected_face_resized)
            self.age_model.setInput(detected_face_blob)
            age_result = self.age_model.forward()
            apparent_age = round(np.sum(age_result[0] * self.age_indexes))

            region = AgePredictionResponse.AgePrediction.Region(int(x), int(y), int(w), int(h))
            age_prediction = AgePredictionResponse.AgePrediction(apparent_age, region, None)
            predictions.append(age_prediction)

        return AgePredictionResponse(predictions)
