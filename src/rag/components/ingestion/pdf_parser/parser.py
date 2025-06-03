from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from openparse import DocumentParser
from openparse.schemas import ParsedDocument
from tqdm import tqdm

from src.rag.components.shared.io import IOManager
from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class MyDocumentParser:
	"""This is my custom document parser that will use pypdf parser to extract text from PDF files."""

	def __init__(
		self,
		io_manager: IOManager,
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
		self.io_manager = io_manager
		self.failed_documents: List[Path] = []

	def parse_document(self, document_path: Path) -> Optional[ParsedDocument]:
		"""
		Parse a single document and save the parsed nodes to the file system.

		Args:
		    document_path (Path): The path to the document to be parsed.
		"""
		try:
			parsed_basic_doc = self.parser.parse(document_path)
			nodes = Node.from_parsed_document(
				parsed_doc=parsed_basic_doc, document_path=document_path
			)
			self.io_manager.save_parsed_nodes(nodes)
			return parsed_basic_doc
		except Exception as e:
			logger.error(f"Error parsing {document_path}: {e}")
			self.failed_documents.append(document_path)

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

	def save_failed_documents(self, output_path: Path):
		"""
		Save the list of failed documents to a file.

		Args:
		    output_path (Path): The path where the failed documents will be saved.
		"""
		if self.failed_documents:
			self.io_manager.write_object_to_file(
				self.io_manager.output_document_path.joinpath("failed_documents.txt"),
				self.failed_documents,
			)
		else:
			logger.info("No failed documents to save.")
