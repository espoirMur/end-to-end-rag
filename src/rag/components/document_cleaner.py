
from haystack import Document
from typing import List, Optional, Generator, Set, Union
from copy import deepcopy
from haystack.nodes import PreProcessor
import re
from unicodedata import normalize as unicode_normalize


class CustomCleaner(PreProcessor):
    def __init__(self, custom_preprocessor=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_preprocessor = custom_preprocessor

    def clean(
        self,
        document: Union[dict, Document],
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        remove_substrings: List[str],
        id_hash_keys: Optional[List[str]] = None,
    ) -> Document:
        """

        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().
        """
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        if isinstance(document, dict):
            document = Document.from_dict(document, id_hash_keys=id_hash_keys)

        # Mainly needed for type checking
        if not isinstance(document, Document):
            raise ValueError(
                "Document must not be of type 'dict' but of type 'Document'.")
        text = document.content
        text = self.custom_preprocessor(text)
        if clean_header_footer:
            text = self._find_and_remove_header_footer(
                text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
            )

        if clean_whitespace:
            lines = text.splitlines()

            cleaned_lines = []
            for line in lines:
                line = line.strip()
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines)

        if clean_empty_lines:
            text = re.sub(r"\n\n+", "\n\n", text)

        for substring in remove_substrings:
            text = text.replace(substring, "")

        if text != document.content:
            document = deepcopy(document)
            document.content = text
        document.content = self.pre

        return document


def pre_clean_document(text) -> str:
    """pre clean the document by removing the accents and replacing the point with the wwt.www with space point before tokenizing the document .
    TOdos : this may have a a downside when the point is in the middle of a words
    and any other side of cleaning that we want to do .
    Args:
        document (_type_): _description_
    """
    result = re.sub(r"This post has already been read \d+ times!",
                    "", text)  # remove unwanted text
    result = unicode_normalize("NFKD", result)
    return result
