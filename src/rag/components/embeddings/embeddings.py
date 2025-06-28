from typing import List

from injector import inject, singleton
from openparse.schemas import ParsedDocument
from sentence_transformers import SentenceTransformer

from src.api.schemas import ModelName
from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("computing embeddings")


@singleton
class EmbeddingComputer:
	@inject
	def __init__(self, model_name: ModelName) -> None:
		self.model_name = model_name
		self.init_model()

	def init_model(self):
		if getattr(self, "model", None):
			logger.info("model is already initialized")
		else:
			logger.info("initializing model")
			logger.info(f"model name: {self.model_name}")
			model = SentenceTransformer(self.model_name)
			self.model = model
			logger.info("model initialized")

	def compute_single_text_embedding(self, text: str) -> List[float]:
		"""compute the embedding of a single text"""
		embedding = self.model.encode(
			[text],
			convert_to_tensor=False,
			show_progress_bar=True,
			normalize_embeddings=True,
		)
		return embedding.tolist()

	def collect_node_text(self, nodes: List[Node]) -> List[str]:
		"""Collect all nodes with text from the nodes."""
		all_text = []
		all_text = [f"passage: {node.text}" for node in nodes]
		return all_text

	def compute_embeddings_in_batch(
		self, all_nodes: List[Node], batch_size: int
	) -> List[Node]:
		"""Process nodes in batches and compute embeddings. Return a list of nodes with embeddings."""
		all_embeddings = []
		print(f"I am dealing with {len(all_nodes)} nodes")
		for i in range(0, len(all_nodes), batch_size):
			batch_nodes = all_nodes[i : i + batch_size]
			texts = self.collect_node_text(batch_nodes)
			batch_embedding = self.model.encode(
				texts,
				convert_to_tensor=False,
				show_progress_bar=True,
				normalize_embeddings=True,
			)
			all_embeddings.extend(batch_embedding)
		nodes_with_embedding = self.assign_embeddings(all_nodes, all_embeddings)
		logger.info(f"Computed embeddings for {len(nodes_with_embedding)} nodes")
		return nodes_with_embedding

	def assign_embeddings(
		self, nodes: List[Node], embeddings: List[List[float]]
	) -> List[ParsedDocument]:
		"""Assign embeddings to nodes"""
		for node, embedding in zip(nodes, embeddings):
			node.embedding = embedding.tolist()
		return nodes
