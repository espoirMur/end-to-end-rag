import os
import shutil
import time
from pathlib import Path

from src.rag.components.ingestion.pdf_parser.parser import MyDocumentParser
from src.rag.components.shared.io import IOManager
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")

# main script that goes through all the pdf files in the document folder and parse them.
if __name__ == "__main__":
	documents_path = Path.home()
	books_path = documents_path.joinpath("Documents")
	file_names = books_path.glob("**/*.pdf")
	output_path = Path.cwd().joinpath("datasets", "parsed_documents")
	try:
		output_path.mkdir(parents=True, exist_ok=False)
	except OSError:
		shutil.rmtree(output_path)
		output_path.mkdir(parents=True, exist_ok=False)

	io_manager = IOManager(input_document_path=books_path, output_path=output_path)
	max_workers = os.cpu_count()
	my_parser = MyDocumentParser(io_manager=io_manager, document_parser_kwargs={})
	start_time = time.time()
	file_names = list(file_names)
	my_parser.parse_documents_parallel(file_names, max_workers=max_workers)
	end_time = time.time()

	logger.info(
		f"Time taken to parse {len(my_parser.io_manager.all_documents)} documents is : {(end_time - start_time)/60} minutes"
	)
