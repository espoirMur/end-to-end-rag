from typing import Dict, List

from injector import inject, singleton
from pymilvus import DataType, MilvusClient
from pymilvus.orm.collection import CollectionSchema

from rag.components.shared.databases.settings import MilvusSettings
from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("milvus_database")


@singleton
class MilvusDatabase:
	@inject
	def __init__(self, milvus_settings: MilvusSettings):
		"""This initializes the Milvus database client."""
		self.settings = milvus_settings
		self.uri = self.settings.uri
		self.token = self.settings.token
		self.vector_dimension = self.settings.vector_dimension
		self.collection_name = self.settings.collection_name
		self.embedding_field_name = "embeddings"

		self.connect()

	def connect(self):
		logger.info("Connecting to Milvus...")
		try:
			self.client = MilvusClient(uri=self.uri, token=self.token)

			logger.info("Connected to Milvus successfully.")
		except Exception as e:
			logger.error(f"Failed to connect to Milvus: {str(e)}")
			raise

	def create_index_if_not_exists(self):
		if not getattr(self, "client", None):
			self.connect()

		collection_name = self.collection_name

		if self.client.has_collection(collection_name):
			logger.info(f"Collection '{collection_name}' already exists.")
		else:
			schema = self.create_schema()
			index_params = self.client.prepare_index_params()
			index_params.add_index(
				field_name=self.embedding_field_name,
				index_type="HNSW",
				metric_type="COSINE",
				params={"nlist": 128},
			)
			self.client.create_collection(
				collection_name=self.collection_name,
				schema=schema,
				index_params=index_params,
			)

			logger.info(f"Created collection '{collection_name}'.")

	def write_data(self, data: List[Dict]):
		logger.info("Writing embeddings to Milvus...")

		if self.client is None:
			self.create_index_if_not_exists()
		try:
			insert_result = self.client.insert(
				collection_name=self.collection_name, data=data
			)
			logger.info(
				f"Successfully inserted {insert_result['insert_count']} entities into Milvus."
			)
		except Exception as e:
			logger.error(f"Failed to insert entities into Milvus: {str(e)}")
			raise

		logger.info("Completed writing embeddings to Milvus.")

	def create_schema(self) -> CollectionSchema:
		"""Create a milvus Schemas"""
		schema = Node.to_milvus_schema(self.client)
		return schema

	def delete_collection(self):
		if self.client is None:
			self.connect()

		collection_name = self.collection_name

		if self.client.has_collection(collection_name):
			self.client.drop_collection(collection_name)
			logger.info(f"Deleted collection '{collection_name}'.")
		else:
			logger.info(f"Collection '{collection_name}' does not exist.")

	def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
		if self.client is None:
			self.connect()

		if not self.client.has_collection(self.collection_name):
			logger.error(f"Collection '{self.collection_name}' does not exist.")
			return []

		try:
			results = self.client.search(
				collection_name=self.collection_name,
				data=query_vector,
				limit=top_k,
				output_fields=Node.keys(),
				params={"metric_type": "COSINE"},
			)
			return results
		except Exception as e:
			logger.error(f"Failed to search in Milvus: {str(e)}")
			return []

	def to_milvus_schema(self) -> CollectionSchema:
		"""Create a milvus schemas for the node class."""
		schema = self.milvus_client.create_schema(
			auto_id=False,
			enable_dynamic_field=True,
		)

		# Add fields to schema
		schema.add_field(
			field_name="node_id",
			datatype=DataType.VARCHAR,
			is_primary=True,
			max_length=100,
		)
		schema.add_field(
			field_name="embeddings",
			datatype=DataType.FLOAT_VECTOR,
			dim=1024,  # addjust according to your embedding dimension
		)
		schema.add_field(
			field_name="variant",
			datatype=DataType.ARRAY,
			max_length=100,
			element_type=DataType.VARCHAR,
			max_capacity=2,
		)
		schema.add_field(
			field_name="tokens",
			datatype=DataType.INT64,
		)
		schema.add_field(field_name="bbox", datatype=DataType.JSON)
		schema.add_field(
			field_name="text",
			datatype=DataType.VARCHAR,
			max_length=65530,
		)
		schema.add_field(
			field_name="object",
			datatype=DataType.VARCHAR,
			max_length=50,
		)
		schema.add_field(
			field_name="score",
			datatype=DataType.FLOAT,
		)
		schema.add_field(
			field_name="previous_texts",
			datatype=DataType.JSON,
			nullable=True,
		)
		schema.add_field(field_name="next_texts", datatype=DataType.JSON, nullable=True)
		schema.add_field(field_name="document", datatype=DataType.JSON)
		# Optionally, add metadata if needed
		schema.add_field(field_name="metadata", datatype=DataType.JSON, nullable=False)
		return schema
