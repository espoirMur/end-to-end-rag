from os import getenv
from typing import Dict, List
from urllib.parse import quote

from dotenv import load_dotenv
from psycopg import Connection, connect

from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("postgres_ingestion_pipeline")


def validate_config():
	required_vars = [
		"POSTGRES_USER",
		"POSTGRES_PASSWORD",
		"POSTGRES_HOST",
		"POSTGRES_PORT",
		"POSTGRES_DB",
	]
	for var in required_vars:
		if not getenv(var):
			raise ValueError(f"Missing required environment variable: {var}")


def create_postgres_connection_uri() -> str:
	"""
	Create a connection to the PostgreSQL database.
	This is a placeholder function. Replace with actual connection logic.
	"""
	load_dotenv()
	validate_config()
	database_user = getenv("POSTGRES_USER")
	database_password = getenv("POSTGRES_PASSWORD")
	database_host = getenv("POSTGRES_HOST")
	database_port = getenv("POSTGRES_PORT")
	database_name = getenv("POSTGRES_DB")
	postgres_uri = f"postgresql://{database_user}:{quote(database_password)}@{database_host}:{database_port}/{database_name}"
	return postgres_uri


def create_postgres_connection(connection_uri: str) -> Connection:
	"""
	Create a connection to the PostgreSQL database.
	This is a placeholder function. Replace with actual connection logic.
	"""
	try:
		database_connection = connect(
			conninfo=connection_uri,
			autocommit=True,
		)
		return database_connection
	except Exception as e:
		logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
		raise


def extract_documents_from_nodes(nodes: List[Node]) -> Dict:
	"""Extract documents from nodes and ensure unique doc_id."""
	documents = {}
	for node in nodes:
		document_dict = node.document.model_dump()
		if document_dict["doc_id"] not in documents:
			documents[document_dict["doc_id"]] = document_dict
	return documents
