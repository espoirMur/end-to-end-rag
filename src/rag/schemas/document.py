import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from openparse.schemas import ImageElement, NodeVariant
from pydantic import BaseModel, Field, computed_field


class BoundingBox(BaseModel):
	page: int
	page_height: float
	page_width: float
	x0: float
	y0: float
	x1: float
	y1: float


class Node(BaseModel):
	embedding: Optional[List[float]] = None
	node_id: str
	variant: List[str]
	tokens: int
	bbox: List[BoundingBox]
	text: str
	elements: tuple

	@computed_field  # type: ignore
	def images(self) -> List[ImageElement]:
		return [e for e in self.elements if e.variant == NodeVariant.IMAGE]


class Document(BaseModel):
	id_: str = Field(
		default_factory=lambda: str(uuid.uuid4()),
		description="Unique ID of the node.",
		exclude=True,
	)
	nodes: List[Node]
	filename: str
	num_pages: int
	coordinate_system: str
	table_parsing_kwargs: Optional[Dict[str, Any]] = None
	last_modified_date: datetime
	last_accessed_date: datetime
	creation_date: datetime
	file_size: int

	def convert_to_milvus(self) -> List[Dict[str, Any]]:
		"""Convert each node in the document to a dictionary format suitable for Milvus."""
		entities = []
		for node in self.nodes:
			entity = {
				"id": node.node_id,
				"embeddings": node.embedding,
				"metadata": {
					"filename": self.filename,
					"num_pages": self.num_pages,
					"coordinate_system": self.coordinate_system,
					"last_modified_date": self.last_modified_date.isoformat(),
					"last_accessed_date": self.last_accessed_date.isoformat(),
					"creation_date": self.creation_date.isoformat(),
					"file_size": self.file_size,
				},
				"text": node.text,
			}
			entities.append(entity)
		return entities
