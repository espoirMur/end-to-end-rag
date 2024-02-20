from pydantic import BaseModel, ValidationError, validator
from typing import List


class InputVector(BaseModel):
    """Input vector"""
    vector: List[float]

    @validator('vector')
    def check_vector_length(self, v):
        """check the vector length
        """
        if len(v) != 1024:
            raise ValueError('Vector must have exactly 1024 elements')
        return v
