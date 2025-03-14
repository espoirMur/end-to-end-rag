import json
from pathlib import Path
from typing import List

from dateutil import parser
from openparse.schemas import ParsedDocument

from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class IOManager:
	"""class for saving and loading documents"""

	def __init__(self, output_path: str):
		self.output_path = Path(output_path)

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

	def load_document(self, document_path: Path) -> ParsedDocument:
		json_string = document_path.read_text()
		doc_dict = json.loads(json_string)
		for node in doc_dict["nodes"]:
			node["elements"] = ()
		doc_dict["last_modified_date"] = parser.parse(
			doc_dict["last_modified_date"]
		).date()
		doc_dict["creation_date"] = parser.parse(doc_dict["creation_date"]).date()
		doc_dict["last_accessed_date"] = parser.parse(
			doc_dict["last_accessed_date"]
		).date()
		parsed_document = ParsedDocument.model_validate(doc_dict, strict=True)
		return parsed_document

	def save_parsed_documents(self, parsed_documents: List[ParsedDocument]):
		"""
		Save parsed documents to the output path.

		Args:
		    parsed_documents (List[ParsedDocument]): The parsed documents to be saved.
		"""
		for parsed_document in parsed_documents:
			self.save_parsed_document(parsed_document)
