from typing import Literal
from pydantic import BaseModel, Field


class ExperimentRequest(BaseModel):
    image: str = Field(description="Base64 encoded RGB image data URL")
    attack: str = "PGD"
    epsilon: float = Field(default=8 / 255, ge=0, le=1)
    alpha: float = Field(default=2 / 255, gt=0, le=1)
    steps: int = Field(default=10, ge=1, le=500)
    random_start: bool = True
    kappa: float = Field(default=0.0, ge=0, le=100)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    c: float = Field(default=1.0, gt=0, le=10000)
    threshold: float | None = Field(default=None, ge=0, le=1)
    purify: bool = True


class ClassifyRequest(BaseModel):
    image: str
    model_id: Literal["victim", "protected"] = "victim"
