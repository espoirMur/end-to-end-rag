import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic_core import ValidationError

from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class IOManager:
	"""class for saving and loading documents"""

	def __init__(
		self,
		input_document_path: Union[Path, str],
		output_path: Union[Path, str],
		glob: str = "**/*.json",
	):
		self.input_document_path = input_document_path
		self.output_document_path = output_path
		self.all_documents = self.input_document_path.glob(glob)
		self.all_documents = list(self.all_documents)
		self.failed_documents: List[Path] = []

	@property
	def number_of_documents(self) -> int:
		"""
		Get the number of documents in the document path.

		Returns:
		    int: The number of documents.
		"""
		return len(self.all_documents)

	def save_parsed_node(self, node: Node, output_path: Path):
		"""
		Save a parsed node to the output path.
		Args:
		        node (Node): The node to be saved.
		        filename (str): The base filename to use for saving.
		"""
		if node is None:
			logger.warning("Skipping None node")
			return
		output_file = output_path.joinpath(
			f"{node.document.filename}_{node.node_id}.json"
		)
		self.write_object_to_file(output_file, node.model_dump_json())

	def write_object_to_file(self, file_path: Path, content):
		"""
		Write the given content to the specified file path.
		Args:
		    file_path (Path): The path to the file where content will be written.
		    content: The content to write to the file. Can be a string, dict, or list.
		"""
		if isinstance(content, (dict, list)):
			content_str = json.dumps(content, indent=2)
		else:
			content_str = str(content)
		with open(file_path, "w") as f:
			f.write(content_str)
			logger.info(f"Saved content to {file_path}")

	def load_node_from_path(self, document_path: Path) -> Optional[Node]:
		"""Load a node object form a Json string"""
		json_string = document_path.read_text()
		try:
			node = Node.model_validate_json(json_string)
			return node
		except ValidationError as e:
			logger.warning(f"Error decoding JSON from {document_path}: {e}")
			self.failed_documents.append(str(document_path))
			return None

	def load_nodes_document(self, start_index: int, end_index: int) -> List[Node]:
		"""Load a list of the nodes from the input path

		Args:
		    start_index (int): The starting index of the documents to load.
		    end_index (int): The ending index of the documents to load.
		Returns:
		    List[CleanedDocument]: A list of parsed documents.
		"""
		nodes = []
		for path in self.all_documents[start_index:end_index]:
			node = self.load_node_from_path(path)
			if node:
				nodes.append(node)
		return nodes

	def save_parsed_nodes(
		self,
		parsed_nodes: List[Node],
		output_folder_name: str = "parsed_documents",
	):
		"""
		Save parsed documents to the output path.

		Args:
		    parsed_documents (List[ParsedDocument]): The parsed documents to be saved.
		"""
		output_path = self.output_document_path.joinpath(output_folder_name)
		output_path.mkdir(parents=True, exist_ok=True)
		for node in parsed_nodes:
			self.save_parsed_node(node, output_path=output_path)
