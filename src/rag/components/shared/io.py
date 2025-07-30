import json
from pathlib import Path
from typing import List, Optional, Union
from uuid import uuid4

import aiofiles
from pydantic_core import ValidationError

from src.rag.schemas.document import Node
from src.shared.logger import setup_logger

logger = setup_logger("io manager")


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
		self.input_document_path = Path(self.input_document_path)
		self.output_document_path = Path(self.output_document_path)
		if not self.input_document_path.exists():
			raise FileNotFoundError(
				f"Input path {self.input_document_path} does not exist."
			)
		self.output_document_path.mkdir(parents=True, exist_ok=True)

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
		logger.info(f"Saved node {node.node_id} to {output_file}")

	def write_object_to_file(self, file_path: Path, content: Union[dict, list, str]):
		"""
		Write the given content to the specified file path.
		Args:
		    file_path (Path): The path to the file where content will be written.
		    content (Union[dict, list, str]): The content to write to the file.
		"""
		if isinstance(content, (dict, list)):
			content_str = json.dumps(content, indent=2)
		else:
			content_str = str(content)
		file_path.write_text(content_str)

	def load_nodes_from_path(self, document_path: Path) -> Optional[List[Node]]:
		"""Load nodes from a json file containing an array of JSON objects."""
		try:
			with open(document_path, "r") as file:
				data = json.load(file)  # Load the entire JSON array

				if not isinstance(data, list):
					logger.warning(
						f"Expected JSON array in {document_path}, got {type(data)}"
					)
					self.failed_documents.append(str(document_path))
					return None

				nodes = []
				for item_string in data:
					try:
						item: dict = json.loads(item_string)
						node = Node.model_validate(item)
						nodes.append(node)
					except ValidationError as e:
						logger.warning(f"Error validating node in {document_path}: {e}")
						self.failed_documents.append(str(document_path))
						continue

				return nodes

		except json.JSONDecodeError as e:
			logger.warning(f"Error decoding JSON from {document_path}: {e}")
			self.failed_documents.append(str(document_path))
			return None
		except Exception as e:
			logger.warning(f"Unexpected error reading {document_path}: {e}")
			self.failed_documents.append(str(document_path))
			return None

	def load_nodes_document(self, start_index: int, end_index: int) -> List[Node]:
		"""Load a list of the nodes from the input path

		Args:
		    start_index (int): The starting index of the documents to load.
		    end_index (int): The ending index of the documents to load.
		Returns:
		    List[Node]: A list of parsed Nodes.
		"""
		all_nodes = []
		for path in self.all_documents[start_index:end_index]:
			nodes = self.load_nodes_from_path(path)
			if nodes:
				all_nodes.extend(nodes)
		return all_nodes

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
		output_path = self.output_document_path
		file_id = str(uuid4())[:8]  # Generate a short unique ID for the file
		all_nodes_json = [node.model_dump_json() for node in parsed_nodes]
		all_nodes_file = output_path.joinpath(
			f"{parsed_nodes[0].document.filename}_{file_id}.json"
		)
		self.write_object_to_file(all_nodes_file, all_nodes_json)
		logger.info(f"Saved {len(parsed_nodes)} nodes to {all_nodes_file}")

	async def save_parsed_nodes_async(
		self,
		parsed_nodes: List[Node],
		filename: str,
	):
		"""
		Save parsed documents to the output path asynchronously.

		Args:
		    parsed_documents (List[ParsedDocument]): The parsed documents to be saved.
		"""
		all_nodes_json = [node.model_dump_json() for node in parsed_nodes]
		all_nodes_file = self.output_document_path.joinpath(f"{filename}.json")
		async with aiofiles.open(all_nodes_file, "w") as f:
			await f.write(json.dumps(all_nodes_json, indent=2))
		logger.info(f"Saved {len(parsed_nodes)} nodes to {all_nodes_file}")

	def save_failed_documents(self, output_path: Path):
		"""
		Save the list of failed documents to a file.

		Args:
		    output_path (Path): The path where the failed documents will be saved.
		"""
		if self.failed_documents:
			self.write_object_to_file(
				output_path,
				self.failed_documents,
			)
		else:
			logger.info("No failed documents to save.")
