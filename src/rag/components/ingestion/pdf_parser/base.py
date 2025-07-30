from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import tqdm
from openparse.schemas import ParsedDocument


class DocumentParserBase(ABC):
	@abstractmethod
	def parse(self, file_path: str):
		"""
		Abstract method to parse a document.

		Args:
		    file_path (str): Path to the document file.

		Returns:
		    Any: Parsed content of the document.
		"""
		pass

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
