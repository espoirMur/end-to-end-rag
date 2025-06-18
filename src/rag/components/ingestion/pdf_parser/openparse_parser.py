from pathlib import Path
from typing import Dict, Optional

from openparse import DocumentParser
from openparse.schemas import ParsedDocument

from src.rag.components.ingestion.pdf_parser.base import DocumentParserBase
from src.rag.components.shared.io import IOManager
from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class MyDocumentParser(DocumentParserBase):
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
			self.io_manager.failed_documents.append(document_path)
