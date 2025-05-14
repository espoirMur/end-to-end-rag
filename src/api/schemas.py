from typing import NewType

from pydantic import BaseModel

ModelName = NewType("ModelName", str)


class Query(BaseModel):
	"""Query model"""

	query: str
