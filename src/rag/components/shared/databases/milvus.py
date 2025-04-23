from typing import Dict, List

from pymilvus import DataType, MilvusClient

from src.shared.logger import setup_logger

logger = setup_logger("milvus_database")


class MilvusDatabase:
	def __init__(self, host, token, vector_dimension, collection_name):
		self.host = host
		self.token = token
		self.vector_dimension = vector_dimension
		self.collection_name = collection_name

	def connect(self):
		logger.info("Connecting to Milvus...")
		try:
			self.client = MilvusClient(uri=self.host, token=self.token)

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
				field_name="embeddings",
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

	def create_schema(self):
		"""Create a milvus Schemas"""
		schema = self.client.create_schema(
			auto_id=False,
			enable_dynamic_field=True,
		)

		# Add fields to schema
		schema.add_field(
			field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
		)
		schema.add_field(
			field_name="embeddings",
			datatype=DataType.FLOAT_VECTOR,
			dim=self.vector_dimension,
		)
		schema.add_field(
			field_name="metadata", datatype=DataType.JSON, is_primary=False
		)
		schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65530)

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
