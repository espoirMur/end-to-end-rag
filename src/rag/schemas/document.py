import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker import BaseChunk
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

	@staticmethod
	def from_docling_document(doc: DoclingDocument, document_path: Path) -> "Document":
		"""Create a Document instance from a Docling ParsedDocument."""
		document_stats = document_path.stat()
		return Document(
			file_path=str(document_path),
			filename=document_path.name,
			num_pages=doc.num_pages(),
			coordinate_system="BOTTOM-LEFT",
			table_parsing_kwargs=None,
			last_modified_date=datetime.fromtimestamp(
				document_stats.st_mtime
			).isoformat(),
			last_accessed_date=datetime.fromtimestamp(
				document_stats.st_atime
			).isoformat(),
			creation_date=datetime.fromtimestamp(document_stats.st_ctime).isoformat(),
			file_size=document_stats.st_size,
		)


class BoundingBox(BaseModel):
	page: int
	page_height: float
	page_width: float
	x0: float
	y0: float
	x1: float
	y1: float


class Node(BaseModel):
	"""Unit of a text document , which can be a chunk of tex, an image or a table."""

	embedding: Optional[List[float]] = None
	node_id: str
	variant: List[str]
	tokens: int
	bbox: List[BoundingBox]
	text: str
	elements: Optional[tuple] = None
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

	def to_milvus_entity(self) -> Dict[str, Any]:
		"""Convert the Node instance to a dictionary suitable for Milvus."""
		return {
			"node_id": self.node_id,
			"embeddings": self.embedding,  # it is embeddings
			"variant": self.variant,
			"tokens": self.tokens,
			"bbox": json.dumps([bbox.model_dump() for bbox in self.bbox]),
			"text": self.text,
			"object": self.object,
			"score": self.score,
			"previous_texts": self.previous_texts,
			"next_texts": self.next_texts,
			"document": self.document.model_dump_json(),
			"metadata": {"test": "this should not contain metadata"},
		}

	@staticmethod
	def docling_chunk_to_node(
		chunker, chunks: Iterator[BaseChunk], document: Document
	) -> List["Node"]:
		"""Convert a Docling chunk to a Node."""
		nodes = []
		for index, chunk in enumerate(chunks):
			previous_texts = [nodes[index - 1].text] if index > 0 else None
			next_texts = [nodes[index + 1].text] if index < len(nodes) - 1 else None
			bounding_box = chunk.meta.doc_items[0].prov[0].model_dump()
			bbox = [
				BoundingBox(
					page=bounding_box.get("page", 0),
					x0=bounding_box.get("bbox").get("l"),
					y0=bounding_box.get("bbox").get("t"),
					x1=bounding_box.get("bbox").get("r"),
					y1=bounding_box.get("bbox").get("b"),
					page_height=bounding_box.get("page_height", 0.0),
					page_width=bounding_box.get("page_width", 0.0),
				)
			]
			enriched_text = chunker.contextualize(chunk)
			previous_texts = [nodes[-1].text] if nodes else None
			next_texts = [chunk.text] if chunk else None

			node = Node(
				node_id=str(uuid.uuid4()),
				variant=["TEXT"],
				tokens=0,
				bbox=bbox,
				text=enriched_text,
				elements=None,
				metadata={},
				document=document,
				previous_texts=previous_texts,
				next_texts=next_texts,
			)
			nodes.append(node)
		return nodes
