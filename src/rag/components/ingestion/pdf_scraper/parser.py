from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from openparse import DocumentParser
from openparse.schemas import ParsedDocument
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class MyDocumentParser:
	"""This is my custom document parser that will use pypdf parser to extract text from PDF files."""

	def __init__(
		self,
		output_path: Path,
		document_parser_kwargs: Dict = {},
		embedding_model: SentenceTransformer = None,
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
		self.embedding_model = embedding_model

	def compute_embeddings(self, document: ParsedDocument) -> ParsedDocument:
		"""
		Computes embeddings for the nodes in the given ParsedDocument using the specified embedding model.

		Args:
		    document (ParsedDocument): The document containing nodes for which embeddings need to be computed.

		Returns:
		    ParsedDocument: The document with computed embeddings for each node.

		Note:
		    If the embedding model is not set (None), the document is returned without any modifications.
		"""
		if self.embedding_model is None:
			return document

		nodes = document.nodes
		node_embedding = self.embedding_model.encode(
			[node.text for node in nodes],
			convert_to_tensor=False,
			show_progress_bar=True,
		)
		for node, embedding in zip(nodes, node_embedding):
			node.embedding = embedding.tolist()
		return document

	def parse_document(self, document_path: Path) -> Optional[ParsedDocument]:
		"""
		Parse a single document, compute the embedding and save it to a path.

		Args:
		    document_path (Path): The path to the document to be parsed.
		"""
		try:
			parsed_basic_doc = self.parser.parse(document_path)
			logger.info(f"Successfully parsed {document_path}")
			parsed_doc_with_embeddings = self.compute_embeddings(parsed_basic_doc)
			self.save_parsed_document(parsed_doc_with_embeddings)
			return parsed_doc_with_embeddings
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

	def save_parsed_document(self, parsed_document: ParsedDocument):
		"""
		Save parsed documents to the output path.

		Args:
		    parsed_document (ParsedDocument): The parsed document to be saved.
		"""
		if parsed_document is None:
			logger.warning("Skipping None document")
			return

		output_file = self.output_path.joinpath(f"{parsed_document.filename}.json")
		with open(output_file, "w") as f:
			f.write(parsed_document.model_dump_json())
		logger.info(f"Saved parsed document to {output_file}")
