from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from openparse import DocumentParser
from openparse.schemas import ParsedDocument
from tqdm import tqdm

from src.shared.logger import setup_logger

logger = setup_logger("data_puller")


class MyDocumentParser:
	"""This is my custom document parser that will use pypdf parser to extract text from PDF files."""

	def __init__(self, output_path: Path, document_parser_kwargs: Dict = {}):
		"""
		Initialize the MyDocumentParser.

		Args:
		    output_path (Path): The path where parsed documents will be saved.
		    document_parser_kwargs (Dict): Additional keyword arguments for the DocumentParser.
		"""
		self.document_parser_kwargs = document_parser_kwargs
		self.parser = DocumentParser(**document_parser_kwargs)
		self.output_path = output_path

	def parse_document(self, document_path: Path) -> Optional[ParsedDocument]:
		"""
		Parse a single document.

		Args:
		    document_path (Path): The path to the document to be parsed.

		Returns:
		    ParsedDocument: The parsed document.
		"""
		try:
			parsed_basic_doc = self.parser.parse(document_path)
			logger.info(f"Successfully parsed {document_path}")
			return parsed_basic_doc
		except Exception as e:
			logger.error(f"Error parsing {document_path}: {e}")
			return None

	def parse_documents_parallel(
		self, file_names, max_workers=4
	) -> List[Optional[ParsedDocument]]:
		"""
		Parse multiple documents in parallel.

		Args:
		    file_names (iterable): An iterable of file paths to be parsed.
		    max_workers (int): The maximum number of worker processes to use.

		Returns:
		    List[ParsedDocument]: A list of parsed documents.
		"""
		results = []
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			# Use tqdm to show progress
			futures = list(
				tqdm(executor.map(self.parse_document, file_names), total=700)
			)

			for future in futures:
				results.append(future)
		return results

	def save_parsed_documents(self, parsed_documents: List[ParsedDocument]):
		"""
		Save parsed documents to the output path.

		Args:
		    parsed_documents (List[ParsedDocument]): A list of parsed documents to be saved.
		"""
		for parsed_document in parsed_documents:
			if parsed_document is None:
				continue
			output_file = self.output_path.joinpath(f"{parsed_document.filename}.json")
			with open(output_file, "w") as f:
				f.write(parsed_document.model_dump_json())
