import asyncio
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import BaseChunker
from tqdm.asyncio import tqdm_asyncio

from src.rag.components.ingestion.pdf_parser.base import DocumentParserBase
from src.rag.components.shared.io import IOManager
from src.rag.schemas.document import Document, Node
from src.shared.logger import setup_logger

logger = setup_logger("pdf_parser")


class DoclingDocumentParser(DocumentParserBase):
	"""This is my custom document parser that will use Docling parser to extract text from PDF files."""

	def __init__(
		self,
		io_manager: IOManager,
		document_parser_kwargs: Dict = {},
		chunker: HybridChunker = HybridChunker(),
	):
		"""
		Initialize the DoclingDocumentParser.

		Args:
		    output_path (Path): The path where parsed documents will be saved.
		    document_parser_kwargs (Dict): Additional keyword arguments for the DocumentParser.
		"""
		self.document_parser_kwargs = document_parser_kwargs
		self.io_manager = io_manager
		self.converter = self.initialize_converter()
		self.chunker = chunker
		self.success_count = 0
		self.failure_count = 0
		self.failed_files: List[Path] = []

	def initialize_converter(self):
		"""
		Initialize the Docling parser with the provided options.
		"""
		accelerator_options = AcceleratorOptions(num_threads=os.cpu_count())

		pdf_pipeline_options = PdfPipelineOptions(
			images_scale=2,
			generate_page_images=True,
			ocr_options=OcrMacOptions(),
			accelerator_options=accelerator_options,
		)
		doc_converter = DocumentConverter(
			format_options={
				InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
			}
		)
		return doc_converter

	async def parse(self, document_path: Path) -> Optional[List[Node]]:
		"""
		Parse a single document asynchronously.

		Args:
		    document_path: Path to the document to parse

		Returns:
		    List of parsed nodes if successful, None otherwise
		"""

		try:
			conversion_result = await asyncio.to_thread(
				self.converter.convert, document_path
			)

			if conversion_result.status != ConversionStatus.SUCCESS:
				self.failure_count += 1
				self.failed_files.append(conversion_result.input.file)
				logger.info(
					f"Document {conversion_result.input.file} failed to convert."
				)
				return None
			else:
				docling_document = conversion_result.document
				chunks = await asyncio.to_thread(self.chunk, document=docling_document)
				nodes = await asyncio.to_thread(
					self.chunk_iter_to_nodes,
					chunks=chunks,
					document=docling_document,
					document_path=document_path,
				)

				# Async save
				await self.io_manager.save_parsed_nodes_async(
					nodes, filename=document_path.name
				)

				self.success_count += 1
				return nodes

		except Exception as e:
			logger.error(f"Error processing {document_path}: {str(e)}", exc_info=True)
			self.failure_count += 1
			self.failed_files.append(document_path)
			return None

	async def parse_documents_async(
		self, file_names: List[Path], max_concurrent_tasks: int = 4
	) -> Tuple[int, int, List[Path]]:
		"""
		Parse multiple documents asynchronously with concurrency control.

		Args:
		    file_names: List of document paths to parse
		    max_concurrent_tasks: Maximum number of concurrent parsing tasks

		Returns:
		    Tuple of (success_count, failure_count, failed_files)
		"""
		# an object that limits the number of concurrent tasks, we can only run max_concurrent tasks
		semaphore = asyncio.Semaphore(max_concurrent_tasks)

		# to ensure only running max concurrent tasks at a time
		async def limited_parse(doc_path):
			async with semaphore:
				return await self.parse(doc_path)

		tasks = [limited_parse(doc_path) for doc_path in file_names]
		await tqdm_asyncio.gather(*tasks, desc="Parsing documents")

		return self.success_count, self.failure_count, self.failed_files

	def chunk(self, document: DoclingDocument) -> Iterator[BaseChunker]:
		"""
		Chunk the parsed document into nodes.

		Args:
		    document (ConversionResult): The parsed document to be chunked.

		Returns:
		    list[Node]: A list of nodes created from the parsed document.
		"""
		chunks = self.chunker.chunk(document)
		return chunks

	def chunk_iter_to_nodes(
		self,
		chunks: Iterator[BaseChunker],
		document: DoclingDocument,
		document_path: Path,
	) -> List[Node]:
		"""convert the chunk iter to nodes"""
		doc_node = Document.from_docling_document(
			doc=document, document_path=document_path
		)
		nodes = Node.docling_chunk_to_node(
			chunker=self.chunker, chunks=chunks, document=doc_node
		)
		return nodes
