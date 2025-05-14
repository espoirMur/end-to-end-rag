from typing import Dict, List

from injector import inject, singleton

from src.rag.components.embeddings.embeddings import EmbeddingComputer
from src.rag.components.shared.databases.milvus import MilvusDatabase
from src.rag.schemas.document import MilvusDocument
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

	def retrieve(self, query: str, top_k: int = 5) -> List[MilvusDocument]:
		"""retrieve the top k documents from the database given the query"""
		query_embedding = self.embedding_computer.compute_single_text_embedding(query)
		retrieved_context = self.milvus_client.search(
			query_vector=query_embedding,
			top_k=top_k,
		)
		post_processed_context = self.post_process_retrieved_context(retrieved_context)
		return post_processed_context

	def post_process_retrieved_context(
		self, retrieved_context: List[Dict]
	) -> List[MilvusDocument]:
		"""post process the retrieved context and return a list of dictionaries with the text , and metadata"""
		document = []
		# 0 because we send only one query
		for result in retrieved_context[0]:
			text = result.entity.get("text")
			distance = result.distance
			id = result.id
			metadata = result.entity.get("metadata")
			milvus_document = MilvusDocument(
				text=text, distance=distance, id=id, metadata=metadata
			)
			document.append(milvus_document)
		return document
