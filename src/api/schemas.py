from pydantic import BaseModel


class Query(BaseModel):
	"""Query model"""

	query: str
