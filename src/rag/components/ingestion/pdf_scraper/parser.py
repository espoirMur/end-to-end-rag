import html
import io
from datetime import datetime
from threading import Lock
from typing import Any, Iterator, Literal, Optional, Union

import numpy as np
import pymupdf
from pymupdf.table import (
	DEFAULT_JOIN_TOLERANCE,
	DEFAULT_MIN_WORDS_HORIZONTAL,
	DEFAULT_MIN_WORDS_VERTICAL,
	DEFAULT_SNAP_TOLERANCE,
)

from src.rag.components.ingestion.pdf_scraper.blob import Blob, Document

_PARAGRAPH_DELIMITER = [
	"\n\n\n",
	"\n\n",
]  # To insert images or table in the middle of the page.

_JOIN_IMAGES = "\n"
_JOIN_TABLES = "\n"
_DEFAULT_PAGES_DELIMITER = "\n\f"
_FORMAT_IMAGE_STR = "\n\n{image_text}\n\n"

_STD_METADATA_KEYS = {"source", "total_pages", "creationdate", "creator", "producer"}


def _format_inner_image(blob: Blob, content: str, format: str) -> str:
	"""Format the content of the image with the source of the blob.

	blob: The blob containing the image.
	format::
	  The format for the parsed output.
	  - "text" = return the content as is
	  - "markdown-img" = wrap the content into an image markdown link, w/ link
	  pointing to (`![body)(#)`]
	  - "html-img" = wrap the content as the `alt` text of an tag and link to
	  (`<img alt="{body}" src="#"/>`)
	"""
	if content:
		source = blob.source or "#"
		if format == "markdown-img":
			content = content.replace("]", r"\\]")
			content = f"![{content}]({source})"
		elif format == "html-img":
			content = (
				f'<img alt="{html.escape(content, quote=True)} ' f'src="{source}" />'
			)
	return content


class MyPydfParser:
	"""A parser for extracting elements from PDF documents, including text and tables."""

	_lock = Lock()

	def __init__(
		self,
		text_kwargs: Optional[dict[str, Any]] = None,
		extract_images: bool = False,
		*,
		password: Optional[str] = None,
		mode: Literal["single", "page"] = "page",
		pages_delimiter: str = _DEFAULT_PAGES_DELIMITER,
		images_inner_format: Literal["text", "markdown-img", "html-img"] = "text",
		extract_tables: Union[Literal["csv", "markdown", "html"], None] = None,
		extract_tables_settings: Optional[dict[str, Any]] = None,
		images_parser: Optional["MyPydfParser"] = None,
	) -> None:
		"""Initialize a parser based on PyMuPDF.

		Args:
		    password: Optional password for opening encrypted PDFs.
		    mode: The extraction mode, either "single" for the entire document or "page"
		        for page-wise extraction.
		    pages_delimiter: A string delimiter to separate pages in single-mode
		        extraction.
		    extract_images: Whether to extract images from the PDF.
		    images_parser: Optional image blob parser.
		    images_inner_format: The format for the parsed output.
		        - "text" = return the content as is
		        - "markdown-img" = wrap the content into an image markdown link, w/ link
		        pointing to (`![body)(#)`]
		        - "html-img" = wrap the content as the `alt` text of an tag and link to
		        (`<img alt="{body}" src="#"/>`)
		    extract_tables: Whether to extract tables in a specific format, such as
		        "csv", "markdown", or "html".
		    extract_tables_settings: Optional dictionary of settings for customizing
		        table extraction.

		Returns:
		    This method does not directly return data. Use the `parse` or `lazy_parse`
		    methods to retrieve parsed documents with content and metadata.

		Raises:
		    ValueError: If the mode is not "single" or "page".
		    ValueError: If the extract_tables format is not "markdown", "html",
		    or "csv".
		"""
		super().__init__()
		if mode not in ["single", "page"]:
			raise ValueError("mode must be single or page")
		if extract_tables and extract_tables not in ["markdown", "html", "csv"]:
			raise ValueError("mode must be markdown")
		self.mode = mode
		self.pages_delimiter = pages_delimiter
		self.password = password
		self.text_kwargs = text_kwargs or {}
		self.extract_images = extract_images
		self.images_inner_format = images_inner_format
		self.extract_tables = extract_tables
		self.extract_tables_settings = extract_tables_settings
		self.images_parser = images_parser

	def lazy_parse(self, blob: Blob) -> Document:
		return self.__lazy_parse(blob)

	def __lazy_parse(self, blob: Blob) -> Iterator[Document]:
		text_kwargs = self.text_kwargs
		self.extract_tables_settings = {
			# See https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables
			"clip": None,
			"vertical_strategy": "lines",
			"horizontal_strategy": "lines",
			"vertical_lines": None,
			"horizontal_lines": None,
			"snap_tolerance": DEFAULT_SNAP_TOLERANCE,
			"snap_x_tolerance": None,
			"snap_y_tolerance": None,
			"join_tolerance": DEFAULT_JOIN_TOLERANCE,
			"join_x_tolerance": None,
			"join_y_tolerance": None,
			"edge_min_length": 3,
			"min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
			"min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
			"intersection_tolerance": 3,
			"intersection_x_tolerance": None,
			"intersection_y_tolerance": None,
			"text_tolerance": 3,
			"text_x_tolerance": 3,
			"text_y_tolerance": 3,
			"strategy": None,  # offer abbreviation
			"add_lines": None,  # optional user-specified lines
		}
		with MyPydfParser._lock:
			with blob.as_bytes_io() as file_path:
				if blob.data is None:
					doc = pymupdf.open(file_path)
				else:
					doc = pymupdf.open(stream=file_path, filetype="pdf")
				if doc.is_encrypted:
					raise ValueError("The PDF is encrypted.")
				doc_metadata = self._extract_metadata(doc, blob=blob)
				full_content = []
				for page in doc:
					all_text = self._get_page_content(doc, page, text_kwargs).strip()
					# Chunking should happen here
					if self.mode == "page":
						yield Document(
							page_content=all_text,
							metadata=_validate_metadata(
								doc_metadata | {"page": page.number}
							),
						)
					else:
						yield Document(
							page_content=self.pages_delimiter.join(full_content),
							metadata=_validate_metadata(doc_metadata),
						)

	def _extract_images_from_page(
		self, doc: pymupdf.Document, page: pymupdf.Page
	) -> str:
		"""Extract images from a PDF page and get the text using images_to_text.

		Args:
		    doc: The PyMuPDF document object.
		    page: The PyMuPDF page object.

		Returns:
		    str: The extracted text from the images on the page.
		"""
		if not self.images_parser:
			return ""

		img_list = page.get_images()
		images = []
		for img in img_list:
			if self.images_parser:
				xref = img[0]
				pix = pymupdf.Pixmap(doc, xref)
				image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
					pix.height, pix.width, -1
				)
				image_bytes = io.BytesIO()
				np.save(image_bytes, image)
				blob = Blob.from_data(
					image_bytes.getvalue(), mime_type="application/x-npy"
				)
				image_text = next(self.images_parser.lazy_parse(blob)).page_content

				images.append(
					_format_inner_image(blob, image_text, self.images_inner_format)
				)
		return _FORMAT_IMAGE_STR.format(
			image_text=_JOIN_IMAGES.join(filter(None, images))
		)

	def _get_page_content(
		self,
		doc: pymupdf.Document,
		page: pymupdf.Page,
		text_kwargs: dict[str, Any],
	) -> str:
		"""Get the text of the page using PyMuPDF and RapidOCR and issue a warning
		if it is empty.

		Args:
		    doc: The PyMuPDF document object.
		    page: The PyMuPDF page object.
		    blob: The blob being parsed.

		Returns:
		    str: The text content of the page.
		"""
		text_from_page = page.get_text(**{**self.text_kwargs, **text_kwargs})
		images_from_page = self._extract_images_from_page(doc, page)
		tables_from_page = self._extract_tables_from_page(page)
		extras = []
		if images_from_page:
			extras.append(images_from_page)
		if tables_from_page:
			extras.append(tables_from_page)
		all_text = _merge_text_and_extras(extras, text_from_page)

		return all_text

	def _extract_tables_from_page(self, page: pymupdf.Page) -> str:
		"""Extract tables from a PDF page.

		Args:
		    page: The PyMuPDF page object.

		Returns:
		    str: The extracted tables in the specified format.
		"""
		if self.extract_tables is None:
			return ""

		tables_list = list(
			pymupdf.table.find_tables(page, **self.extract_tables_settings)
		)
		if tables_list:
			if self.extract_tables == "markdown":
				return _JOIN_TABLES.join([table.to_markdown() for table in tables_list])
			elif self.extract_tables == "html":
				return _JOIN_TABLES.join(
					[
						table.to_pandas().to_html(
							header=False,
							index=False,
							bold_rows=False,
						)
						for table in tables_list
					]
				)
			elif self.extract_tables == "csv":
				return _JOIN_TABLES.join(
					[
						table.to_pandas().to_csv(
							header=False,
							index=False,
						)
						for table in tables_list
					]
				)
			else:
				raise ValueError(
					f"extract_tables {self.extract_tables} not implemented"
				)
		return ""

	def _extract_metadata(self, doc: pymupdf.Document, blob: Blob) -> dict:
		"""Extract metadata from the document and page.

		Args:
		    doc: The PyMuPDF document object.
		    blob: The blob being parsed.

		Returns:
		    dict: The extracted metadata.
		"""
		return _purge_metadata(
			dict(
				{
					"producer": "PyMuPDF",
					"creator": "PyMuPDF",
					"creationdate": "",
					"source": blob.source,  # type: ignore[attr-defined]
					"file_path": blob.source,  # type: ignore[attr-defined]
					"total_pages": len(doc),
				},
				**{
					k: doc.metadata[k]
					for k in doc.metadata
					if isinstance(doc.metadata[k], (str, int))
				},
			)
		)


def _purge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
	"""Purge metadata from unwanted keys and normalize key names.

	Args:
	    metadata: The original metadata dictionary.

	Returns:
	    The cleaned and normalized the key format of metadata dictionary.
	"""
	new_metadata: dict[str, Any] = {}
	map_key = {
		"page_count": "total_pages",
		"file_path": "source",
	}
	for k, v in metadata.items():
		if type(v) not in [str, int]:
			v = str(v)
		if k.startswith("/"):
			k = k[1:]
		k = k.lower()
		if k in ["creationdate", "moddate"]:
			try:
				new_metadata[k] = datetime.strptime(
					v.replace("'", ""), "D:%Y%m%d%H%M%S%z"
				).isoformat("T")
			except ValueError:
				new_metadata[k] = v
		elif k in map_key:
			# Normalize key with others PDF parser
			new_metadata[map_key[k]] = v
			new_metadata[k] = v
		elif isinstance(v, str):
			new_metadata[k] = v.strip()
		elif isinstance(v, int):
			new_metadata[k] = v
	return new_metadata


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
	"""Validate that the metadata has all the standard keys and the page is an integer.

	The standard keys are:
	- source
	- total_page
	- creationdate
	- creator
	- producer

	Validate that page is an integer if it is present.
	"""

	if not _STD_METADATA_KEYS.issubset(metadata.keys()):
		raise ValueError("The PDF parser must valorize the standard metadata.")
	if not isinstance(metadata.get("page", 0), int):
		raise ValueError("The PDF metadata page must be a integer.")
	return metadata


def _merge_text_and_extras(extras: list[str], text_from_page: str) -> str:
	"""Insert extras such as image/table in a text between two paragraphs if possible,
	else at the end of the text.

	Args:
	    extras: List of extra content (images/tables) to insert.
	    text_from_page: The text content from the page.

	Returns:
	    The merged text with extras inserted.
	"""

	def _recurs_merge_text_and_extras(
		extras: list[str], text_from_page: str, recurs: bool
	) -> Optional[str]:
		if extras:
			for delim in _PARAGRAPH_DELIMITER:
				pos = text_from_page.rfind(delim)
				if pos != -1:
					# search penultimate, to bypass an error in footer
					previous_text = None
					if recurs:
						previous_text = _recurs_merge_text_and_extras(
							extras, text_from_page[:pos], False
						)
					if previous_text:
						all_text = previous_text + text_from_page[pos:]
					else:
						all_extras = ""
						str_extras = "\n\n".join(filter(lambda x: x, extras))
						if str_extras:
							all_extras = delim + str_extras
						all_text = (
							text_from_page[:pos] + all_extras + text_from_page[pos:]
						)
					break
			else:
				all_text = None
		else:
			all_text = text_from_page
		return all_text

	all_text = _recurs_merge_text_and_extras(extras, text_from_page, True)
	if not all_text:
		all_extras = ""
		str_extras = "\n\n".join(filter(lambda x: x, extras))
		if str_extras:
			all_extras = _PARAGRAPH_DELIMITER[-1] + str_extras
		all_text = text_from_page + all_extras

	return all_text
