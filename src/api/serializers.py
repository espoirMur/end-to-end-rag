from pydantic import BaseModel, ValidationError, field_validator
from typing import List


class InputVector(BaseModel):
    """Input vector"""
    vector: List[float]

    @field_validator('vector', mode='before')
    def check_vector_length(cls, v):
        """check the vector length
        """
        if len(v) != 1024:
            raise ValidationError('Vector must have exactly 1024 elements')
        return v
