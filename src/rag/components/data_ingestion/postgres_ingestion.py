# The Main script to insert the code in The postgres database
import argparse
from os import getenv
from pathlib import Path

from tqdm import trange

from src.rag.components.data_ingestion.utils import (
	create_postgres_connection,
	create_postgres_connection_uri,
	extract_documents_from_nodes,
)
from src.rag.components.shared.databases.postgres import PostgresVectorDBClient
from src.rag.components.shared.io import IOManager
from src.rag.schemas.document import Document, Node
from src.shared.logger import setup_logger

logger = setup_logger("postgres_ingestion_pipeline")

DEFAULT_COLLECTION_NAME = "my_documents"
EMBEDDING_DIMENSION = int(getenv("EMBEDDING_DIMENSION", 1024))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Process the argument to connect to the database."
	)
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=100,
		help="Number of documents to load at a time.",
	)

	# document path argument
	parser.add_argument(
		"--document_path",
		type=str,
		default=Path.cwd().joinpath("datasets", "parsed_documents_with_embeddings"),
		help="Path to the documents to be processed.",
	)
	parser.add_argument(
		"--collection_name",
		type=str,
		default=DEFAULT_COLLECTION_NAME,
		help="Name of the collection to insert into the database",
	)

	# add number of documents to process
	parser.add_argument(
		"--number_of_documents",
		type=int,
		default=None,
		help="Number of documents to process.",
	)
	args = parser.parse_args()

	input_path = Path(args.document_path)
	# the output path is the same as the input path, as we are not saving any new files
	output_path = Path(args.document_path)
	io_manager = IOManager(input_document_path=input_path, output_path=output_path)

	assert (
		io_manager.input_document_path.exists()
	), f"Document path {io_manager.input_document_path} does not exist."
	connection_uri = create_postgres_connection_uri()
	connection = create_postgres_connection(connection_uri)
	postgres_client = PostgresVectorDBClient(
		connection=connection,
		namespace=args.collection_name,
	)

	postgres_client.create_table(
		name="documents", schema=Document.to_sql_schema(), if_not_exists=True
	)
	postgres_client.create_table(
		name="nodes",
		schema=Node.to_sql_schema(
			embedding_dimension=EMBEDDING_DIMENSION,
			table_prefix=DEFAULT_COLLECTION_NAME,
		),
		if_not_exists=True,
	)
	# drop foreign key if it exists to speed up the insertiion process. This will be added later
	postgres_client.drop_constraint(
		table_name="nodes",
		constraint_name=f"{DEFAULT_COLLECTION_NAME}_nodes_document_id_foreign_key",
	)
	document_to_process = (
		args.number_of_documents
		if args.number_of_documents
		else io_manager.number_of_documents
	)
	logger.info(
		f"Processing {document_to_process} documents in chunks of size {args.chunk_size}"
	)
	for i in trange(
		0, document_to_process, args.chunk_size, desc="Ingesting documents"
	):
		logger.info(f"Processing documents from index {i} to {i + args.chunk_size}")
		nodes = io_manager.load_nodes_document(i, i + args.chunk_size)
		nodes_sql = [doc.to_sql_insert(DEFAULT_COLLECTION_NAME) for doc in nodes]
		documents = extract_documents_from_nodes(nodes)
		try:
			inserted_nodes = postgres_client.bulk_insert(
				table_name="nodes", data=nodes_sql, returning=["node_id"]
			)
			inserted_documents = postgres_client.bulk_insert(
				table_name="documents",
				data=list(documents.values()),
				returning=["doc_id"],
			)

			logger.info(
				f"Finished processing documents from index {i} to {i + args.chunk_size}"
			)
		except Exception as e:
			logger.error(f"Failed to insert entities into postgres {str(e)}")
			raise
	postgres_client.add_foreign_key_to_table(
		table_name="nodes",
		column_name="document_id",
		foreign_table="documents",
		foreign_column="doc_id",
		if_not_exists=False,
	)

	postgres_client.create_index(
		table_name="nodes",
		column_name="embedding",
		index_config="USING vchordrq",
		if_not_exists=True,
	)
	logger.info("Finished processing all documents.")
