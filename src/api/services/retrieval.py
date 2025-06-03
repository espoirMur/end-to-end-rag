from typing import List

from injector import inject, singleton

from src.rag.components.embeddings.embeddings import EmbeddingComputer
from src.rag.components.shared.databases.milvus import MilvusDatabase
from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("retriever_services")


@singleton
class RetrieverServices:
	@inject
	def __init__(
		self, milvus_client: MilvusDatabase, embedding_computer: EmbeddingComputer
	):
		# later we can customize this to handle multiple databases.
		self.milvus_client = milvus_client
		self.embedding_computer = embedding_computer
		logger.info("Retriever services initialized.")

	def search(self, query: str, top_k: int = 5) -> List[Node]:
		"""retrieve the top k documents from the database given the query"""
		query_embedding = self.embedding_computer.compute_single_text_embedding(
			f"query : {query}"
		)
		retrieved_context = self.milvus_client.search(
			query_vector=query_embedding,
			top_k=top_k,
		)[0]
		results = self.post_process_retrieved_context(retrieved_context)
		return results

	def post_process_retrieved_context(
		self, retrieved_context: List[Node]
	) -> List[Node]:
		"""post process the retrieved context and return a list of dictionaries with the text , and metadata"""
		responses = []

		for results in retrieved_context:
			entity = results.entity
			node = Node.from_milvus_entity(entity)
			responses.append(node)
		return responses
