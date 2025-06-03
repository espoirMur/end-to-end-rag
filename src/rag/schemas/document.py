import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openparse.schemas import ImageElement, ParsedDocument
from pydantic import BaseModel, Field


class Document(BaseModel):
	doc_id: str = Field(
		default_factory=lambda: str(uuid.uuid4()),
		description="Unique ID of the node.",
		exclude=True,
	)
	file_path: str
	filename: Optional[str] = (None,)
	num_pages: int
	coordinate_system: str
	table_parsing_kwargs: Optional[Dict[str, Any]] = None
	last_modified_date: datetime
	last_accessed_date: datetime
	creation_date: datetime
	file_size: int
	object: Literal["ingest.document"] = "ingest.document"
	doc_metadata: Optional[Dict[str, Any]] = None

	@staticmethod
	def curate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
		"""Remove unwanted metadata keys."""
		for key in ["doc_id", "window", "original_text"]:
			metadata.pop(key, None)
		return metadata

	def convert_to_milvus(self) -> List[Dict[str, Any]]:
		"""Convert each node in the document to one entity for milvus storage.
		Add the rest of the document metadata to each node's metadata.
		"""
		json_data = self.model_dump(mode="json")
		nodes = json_data.pop("nodes", None)
		for node in nodes:
			node["metadata"] = json_data.copy()
			node["text"] = f"passage: {node['text']}"
		return nodes


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
	object: Literal["context.chunk"] = "context.chunk"
	score: float = 0.0
	previous_texts: Optional[List[str]] = None
	next_texts: Optional[List[str]] = None
	document: Document

	@classmethod
	def keys(cls):
		model_keys = cls.model_fields.keys()
		model_keys = model_keys - {"embedding"}
		return list(model_keys)

	@staticmethod
	def from_milvus_entity(entity: Dict[str, Any]) -> "Node":
		"""Create a DocNode instance from a milvus entity."""
		bbox = [BoundingBox(**bbox) for bbox in entity.get("bbox", [])]
		elements = tuple(ImageElement(**el) for el in entity.get("elements", []))
		return Node(
			node_id=entity["node_id"],
			variant=entity["variant"],
			tokens=entity["tokens"],
			bbox=bbox,
			text=entity["text"],
			elements=elements,
			metadata=entity.get("metadata"),
		)

	@classmethod
	def from_node_with_score(cls, node_with_score) -> "Node":
		"""Create a DocNode from a node with score (for search results)."""
		node = node_with_score.node
		return cls(
			object="context.chunk",
			score=node_with_score.score or 0.0,
			node_id=getattr(node, "node_id", "-"),
			variant=getattr(node, "variant", []),
			tokens=getattr(node, "tokens", 0),
			bbox=getattr(node, "bbox", []),
			text=getattr(node, "text", ""),
			elements=getattr(node, "elements", ()),
			metadata=getattr(node, "metadata", None),
			previous_texts=getattr(node_with_score, "previous_texts", None),
			next_texts=getattr(node_with_score, "next_texts", None),
		)

	@classmethod
	def from_parsed_document(
		cls, parsed_doc: ParsedDocument, document_path: Path
	) -> List["Node"]:
		"""Create multiple nodes instances from a parsed document."""
		nodes = []
		document = Document(
			doc_id=parsed_doc.id_,
			# the path of the document in the file system
			file_path=str(document_path),
			filename=parsed_doc.filename.replace(" ", "_"),
			num_pages=parsed_doc.num_pages,
			coordinate_system=parsed_doc.coordinate_system,
			table_parsing_kwargs=parsed_doc.table_parsing_kwargs,
			last_modified_date=parsed_doc.last_modified_date,
			last_accessed_date=parsed_doc.last_accessed_date,
			creation_date=parsed_doc.creation_date,
			file_size=parsed_doc.file_size,
			doc_metadata={},
		)
		for idx, node in enumerate(parsed_doc.nodes):
			previous_texts = [parsed_doc.nodes[idx - 1].text] if idx > 0 else None
			next_texts = (
				[parsed_doc.nodes[idx + 1].text]
				if idx < len(parsed_doc.nodes) - 1
				else None
			)
			bounding_boxes = [
				BoundingBox(
					page=bbox.page,
					page_height=bbox.page_height,
					page_width=bbox.page_width,
					x0=bbox.x0,
					y0=bbox.y0,
					x1=bbox.x1,
					y1=bbox.y1,
				)
				for bbox in node.bbox
			]
			nodes_to_add = Node(
				node_id=node.node_id,
				variant=node.variant,
				tokens=node.tokens,
				bbox=bounding_boxes,
				text=node.text,
				elements=tuple(node.elements),
				metadata=document.doc_metadata,
				document=document,
				previous_texts=previous_texts,
				next_texts=next_texts,
			)
			nodes.append(nodes_to_add)
		return nodes
