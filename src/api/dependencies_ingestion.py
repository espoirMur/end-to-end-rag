from injector import Injector, singleton
from sentence_transformers import SentenceTransformer

from src.rag.components.data_ingestion.utils import (
	create_postgres_connection,
	create_postgres_connection_uri,
)
from src.rag.components.shared.databases.postgres import PostgresVectorDBClient
from src.shared.logger import setup_logger

logger = setup_logger("application_injector")


def create_application_injector() -> Injector:
	_injector = Injector(auto_bind=True)

	# Need to replace this with the setting from environments
	COLLECTION_NAME = "my_documents"
	embedding_model_name = "intfloat/multilingual-e5-large"

	# is_this right?

	connection_uri = create_postgres_connection_uri()

	connection = create_postgres_connection(connection_uri)
	logger.info("Postgres connection established")
	postgres_client = PostgresVectorDBClient(
		connection=connection, namespace=COLLECTION_NAME
	)
	logger.info(f"Postgres client initialized for collection {COLLECTION_NAME}")
	logger.info("Initializing embedding model")
	embedding_model = SentenceTransformer(embedding_model_name)
	logger.info("Embedding model initialized")
	# so basically, bind what is doing, anywhere something requires a PostgresVectorDBClient, it will use this instance
	_injector.binder.bind(PostgresVectorDBClient, to=postgres_client, scope=singleton)
	# every time something requires an EmbeddingComputer, it will create a new instance
	_injector.binder.bind(SentenceTransformer, to=embedding_model, scope=singleton)
	return _injector


"""
Global injector for the application.

Avoid using this reference, it will make your code harder to test.

Instead, use the `request.state.injector` reference, which is bound to every request
"""
global_injector: Injector = create_application_injector()
