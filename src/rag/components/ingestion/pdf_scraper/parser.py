from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from openparse import DocumentParser
from openparse.schemas import ParsedDocument
from tqdm import tqdm

from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class MyDocumentParser:
	"""This is my custom document parser that will use pypdf parser to extract text from PDF files."""

	def __init__(
		self,
		output_path: Path,
		document_parser_kwargs: Dict = {},
	):
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
		Parse a single document, compute the embedding and save it to a path.

		Args:
		    document_path (Path): The path to the document to be parsed.
		"""
		try:
			parsed_basic_doc = self.parser.parse(document_path)
			logger.info(f"Successfully parsed {document_path}")
			self.save_parsed_document(parsed_basic_doc)
			return parsed_basic_doc
		except Exception as e:
			logger.error(f"Error parsing {document_path}: {e}")
			return None

	def parse_documents_parallel(
		self, file_names: List[Path], max_workers: int = 4
	) -> List[Optional[ParsedDocument]]:
		"""
		Parse multiple documents in parallel.

		Args:
		    file_names (List[Path]): A list of file paths to be parsed.
		    max_workers (int): The maximum number of worker processes to use.

		Returns:
		    List[Optional[ParsedDocument]]: A list of parsed documents.
		"""
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			results = list(
				tqdm(
					executor.map(self.parse_document, file_names), total=len(file_names)
				)
			)
		return results
