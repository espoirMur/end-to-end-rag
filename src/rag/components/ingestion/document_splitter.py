from typing import List
from unicodedata import normalize as unicode_normalize

from haystack.core.component import component
from haystack.dataclasses import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@component
class RecursiveCharacterTextSplitterComponent:
	"""
	A component generating personal welcome message and making it upper case
	"""

	def __init__(
		self, chunk_size: int, chunk_overlap: int, is_separator_regex: bool = False
	):
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.is_separator_regex = is_separator_regex
		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=self.chunk_size,
			chunk_overlap=chunk_overlap,
			is_separator_regex=self.is_separator_regex,
		)

	@component.output_types(documents=List[Document])
	def run(self, texts: List[Document]):
		documents = []
		for text in texts:
			text_normalize_text = unicode_normalize("NFKD", text.content)
			chunk_split_texts = self.splitter.create_documents([text_normalize_text])
			for chunk_split_text in chunk_split_texts:
				chunk_document = Document(
					content=chunk_split_text.page_content, meta={"article_id": text.id}
				)
				documents.append(chunk_document)

		return {"documents": documents}
