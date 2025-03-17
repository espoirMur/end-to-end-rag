import json
from pathlib import Path
from typing import List

from dateutil import parser
from openparse.schemas import ParsedDocument

from src.rag.schemas.document import Document as CleanedDocument
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class IOManager:
	"""class for saving and loading documents"""

	def __init__(self, document_path: str):
		self.document_path = Path(document_path)
		self.all_json_documents = self.document_path.glob("**/*.json")
		self.all_json_documents = list(self.all_json_documents)
		self.output_path = self.document_path.parent.joinpath(
			"parsed_documents_with_embedding"
		)

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

	def load_document(self, document_path: Path) -> CleanedDocument:
		json_string = document_path.read_text()
		try:
			doc_dict = json.loads(json_string)
			for node in doc_dict["nodes"]:
				node.elements = ()
				doc_dict["last_modified_date"] = parser.parse(
					doc_dict["last_modified_date"]
				).date()
				doc_dict["creation_date"] = parser.parse(
					doc_dict["creation_date"]
				).date()
				doc_dict["last_accessed_date"] = parser.parse(
					doc_dict["last_accessed_date"]
				).date()
			cleaned_document = CleanedDocument(**doc_dict)
			return cleaned_document
		except json.JSONDecodeError as e:
			logger.error(f"Error decoding JSON from {document_path}: {e}")
			pass

	def load_documents(self, start_index: int, end_index) -> List[CleanedDocument]:
		documents = []
		for path in self.all_json_documents[start_index:end_index]:
			document = self.load_document(path)
			documents.append(document)
		return documents

	def save_parsed_documents(self, parsed_documents: List[ParsedDocument]):
		"""
		Save parsed documents to the output path.

		Args:
		    parsed_documents (List[ParsedDocument]): The parsed documents to be saved.
		"""
		for parsed_document in parsed_documents:
			self.save_parsed_document(parsed_document)
