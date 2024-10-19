from haystack.core.component import component
from haystack.dataclasses import Document
from copy import deepcopy
from typing import Union, List, Optional
from bertopic import BERTopic
import re
from unicodedata import normalize as unicode_normalize


@component
class CustomCleaner:
    def __init__(self, custom_preprocessor=None, **kwargs):
        self.custom_preprocessor = custom_preprocessor
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clean(
        self,
        document: Union[dict, Document],
        clean_whitespace: bool,
        clean_empty_lines: bool,
        remove_substrings: List[str]
    ) -> Document:
        """

        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().
        """

        if isinstance(document, dict):
            document = Document.from_dict(document)

        # Mainly needed for type checking
        if not isinstance(document, Document):
            raise ValueError(
                "Document must not be of type 'dict' but of type 'Document'.")
        text = document.content
        text = self.custom_preprocessor(text)

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
        document.content = text

        return document

    @component.output_types(documents=List[Document])
    def run(self, texts: List[Document]):
        documents = []
        for doc in texts:
            doc = self.clean(
                document=doc,
                clean_whitespace=True,
                clean_empty_lines=True,
                remove_substrings=[
                    "This post has already been read \d+ times!"],
            )
            documents.append(doc)

        return {"documents": documents}


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
