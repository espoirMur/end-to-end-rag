import argparse
import time
from pathlib import Path

from src.rag.components.embeddings.embeddings import EmbeddingComputer
from src.rag.components.shared.io import IOManager
from src.shared.logger import setup_logger

logger = setup_logger("embeddings computation")


def main(
	embedding_model_name: str,
	input_path: str,
	output_path: str,
	batch_size: int,
	limit: int,
):
	input_path = Path(input_path)
	input_path = Path.cwd().joinpath("datasets", input_path)
	output_path = Path(output_path)

	output_path.mkdir(parents=True, exist_ok=True)
	io_manager = IOManager(input_document_path=input_path, output_path=output_path)
	embedding_computer = EmbeddingComputer(model_name=embedding_model_name)
	# save the starttime here
	start_time = time.time()
	logger.info(f"Start time: {start_time}")
	if limit:
		doc_to_process = io_manager.all_documents[:limit]
	else:
		doc_to_process = io_manager.all_documents
	logger.info(
		f"Number of documents to process: {len(doc_to_process)} from {input_path}"
	)
	for i in range(0, len(doc_to_process), batch_size):
		logger.info(f"Processing documents from index {i} to {i + batch_size}")
		nodes = io_manager.load_nodes_document(i, i + batch_size)
		nodes = embedding_computer.compute_embeddings_in_batch(
			nodes, batch_size=batch_size
		)
		io_manager.save_parsed_nodes(nodes)
		logger.info(f"Finished processing documents from index {i} to {i + batch_size}")
	end_time = time.time()
	elapsed_time = (end_time - start_time) / 60
	logger.info(
		f"Total time taken to compute embedding for {io_manager.number_of_documents} is : {elapsed_time} minutes"
	)
	io_manager.write_object_to_file(
		output_path.joinpath("failed_document_list.txt"), io_manager.failed_documents
	)
	logger.info(
		f"done saving the document with embeddings to the output path {io_manager.output_document_path}"
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Process documents and compute embeddings."
	)
	parser.add_argument(
		"--embedding_model_name",
		type=str,
		default="intfloat/multilingual-e5-large",
		help="Name of the embedding model to use.",
	)
	parser.add_argument(
		"--input_path",
		type=str,
		default="parsed_documents/parsed_documents",
		help="Path to the input documents.",
	)
	parser.add_argument(
		"--output_path",
		type=str,
		default=Path.cwd().joinpath("datasets", "parsed_documents_with_embeddings"),
		help="Path to save the documents with embeddings.",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=50,
		help="Batch size for computing embeddings.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="limit of documents to load. If None, all documents are processed.",
	)
	args = parser.parse_args()

	main(
		embedding_model_name=args.embedding_model_name,
		input_path=args.input_path,
		output_path=args.output_path,
		batch_size=args.batch_size,
		limit=args.limit,
	)
