from dataclasses import dataclass
from typing import List, ForwardRef

AgePrediction = ForwardRef('AgePrediction')
Region = ForwardRef('Region')


@dataclass
class AgePredictionResponse:
    predictions: List[AgePrediction]

    @dataclass
    class AgePrediction:
        age: int
        region: Region
        face_confidence: float | None

        @dataclass
        class Region:
            x: int
            y: int
            w: int
            h: int
