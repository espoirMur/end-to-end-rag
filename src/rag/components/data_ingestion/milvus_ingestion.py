# The Main script to insert the code in The Milvus database
import argparse
from pathlib import Path

from rag.components.shared.databases.settings import MilvusSettings
from src.rag.components.shared.databases.milvus import MilvusDatabase
from src.rag.components.shared.io import IOManager
from src.shared.logger import setup_logger

logger = setup_logger("milvus ingestion pipeline")

if __name__ == "__main__":
	# Initialize the IOManager and MilvusDatabase

	parser = argparse.ArgumentParser(
		description="Process documents and compute embeddings."
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
		default=Path.home().joinpath("datasets", "parsed_documents_with_embeddings"),
		help="Path to the documents to be processed.",
	)
	parser.add_argument(
		"--collection_name",
		type=str,
		default="my_collection",
		help="Name of the collection in Milvus.",
	)
	parser.add_argument(
		"--host",
		type=str,
		default="http://localhost:19530",
		help="Host of the Milvus database.",
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
	settings = MilvusSettings(
		uri=args.host,
		collection_name=args.collection_name,
		vector_dimension=1024,
	)
	milvus_client = MilvusDatabase(milvus_settings=settings)
	milvus_client.create_index_if_not_exists()
	document_to_process = (
		args.number_of_documents
		if args.number_of_documents
		else io_manager.number_of_documents
	)
	logger.info(
		f"Processing {document_to_process} documents in chunks of size {args.chunk_size}"
	)
	for i in range(0, document_to_process, args.chunk_size):
		logger.info(f"Processing documents from index {i} to {i + args.chunk_size}")
		nodes = io_manager.load_nodes_document(i, i + args.chunk_size)
		nodes_to_write = []
		for node in nodes:
			node_json = node.to_milvus_entity()
			nodes_to_write.append(node_json)
		try:
			milvus_client.write_data(nodes_to_write)
			logger.info(
				f"Finished processing documents from index {i} to {i + args.chunk_size}"
			)
		except Exception as e:
			logger.error(f"Failed to insert entities into Milvus: {str(e)}")
			raise
	logger.info("Finished processing all documents.")
