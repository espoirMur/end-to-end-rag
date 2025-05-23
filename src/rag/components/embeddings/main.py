import argparse
from pathlib import Path

from src.rag.components.embeddings.embeddings import EmbeddingComputer
from src.rag.components.shared.io import IOManager
from src.shared.logger import setup_logger

logger = setup_logger("embeddings computation")

documents_path = Path.home()

books_path = documents_path.joinpath("Documents")


embedding_model_name = "intfloat/multilingual-e5-large"

embedding_computer = EmbeddingComputer(model_name=embedding_model_name)
output_path = Path.cwd().joinpath("datasets", "parsed_documents")
io_manager = IOManager(output_path)


if __name__ == "__main__":
	# Chunk document in size of 100, compute the embeddings and save the document with the embeddings
	# later this process can be done in  distribute manner in a Cluster of vertex ai pipeline or Flyte Pipelines
	parser = argparse.ArgumentParser(
		description="Process documents and compute embeddings."
	)
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=100,
		help="Number of documents to load at a time.",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=4,
		help="Batch size for computing embeddings.",
	)
	args = parser.parse_args()

	chunk_size = args.chunk_size
	batch_size = args.batch_size
	for i in range(0, len(io_manager.all_json_documents[:5]), chunk_size):
		logger.info(f"Processing documents from index {i} to {i + chunk_size}")
		documents = io_manager.load_documents(i, i + chunk_size)
		documents = embedding_computer.compute_embeddings(
			documents, batch_size=batch_size
		)
		io_manager.save_parsed_documents(documents)
		logger.info(f"Finished processing documents from index {i} to {i + chunk_size}")
