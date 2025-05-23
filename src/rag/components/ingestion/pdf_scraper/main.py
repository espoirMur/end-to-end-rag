import os
import time
from pathlib import Path

from src.rag.components.ingestion.pdf_scraper.parser import MyDocumentParser
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")

# Parse all documents
if __name__ == "__main__":
	documents_path = Path.home()
	books_path = documents_path.joinpath("Documents")
	file_names = books_path.glob("**/*.pdf")
	output_path = Path.cwd().joinpath("datasets", "parsed_documents")
	output_path.mkdir(parents=True, exist_ok=True)
	max_workers = os.cpu_count()
	my_parser = MyDocumentParser(output_path=output_path)
	start_time = time.time()
	file_names = list(file_names)
	my_parser.parse_documents_parallel(file_names, max_workers=max_workers)
	end_time = time.time()

	print(f"Time taken to parse documents: {end_time - start_time} seconds")
