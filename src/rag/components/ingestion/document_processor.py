import os

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from ingestion.document_splitter import RecursiveCharacterTextSplitterComponent
from shared.database import execute_query, generate_database_connection, postgres_uri
from shared.document_store import MyPgVectorDocumentStore
from sqlalchemy.engine import CursorResult

os.environ["PG_CONN_STR"] = postgres_uri

TABLE_CREATION_STRING = """
CREATE TABLE IF NOT EXISTS article_embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER,
    chunk TEXT,
    chunk_vector VECTOR(768),
    CONSTRAINT fk_article_id FOREIGN KEY (article_id) REFERENCES article(id)
);
"""

INSERT_STATEMENT_STRING = """
INSERT INTO article_embeddings (article_id, chunk, chunk_vector)
values (%(article_id)s, %(chunk)s, %(chunk_vector)s)
"""

UPDATE_STATEMENT_STRING = """
ON CONFLICT (id) DO UPDATE SET
article_id = EXCLUDED.article_id,
chunk = EXCLUDED.chunk,
chunk_vector = EXCLUDED.chunk_vector,
"""


class DocumentProcessor:
	"""
	This class will handle the document processor for the RAG system.

	It will be responsible for ingesting the document chunks into the database.
	"""

	def __init__(self, embedding_model_id: str = "camembert-base") -> None:
		self.text_splitter = self.init_text_splitter()
		self.document_cleaner = self.init_document_cleaner()
		self.document_store = self.init_document_store()
		self.embedding_model_id = embedding_model_id
		self.document_embedder = self.init_document_embedder()
		self.document_writer = self.init_document_writer()

	def read_documents(
		self, table_name: str = "article", use_current_date: bool = True
	) -> CursorResult:
		"""
		Read the documents from the database. and Return a cursor of results
		"""
		query = f"""
            SELECT id, title, content, posted_at, website_origin, url, author
            FROM {table_name}
        """
		if use_current_date:
			query = query + " WHERE posted_at::date = CURRENT_DATE"
		connection = generate_database_connection()

		return execute_query(connection, query)

	def init_document_store(self):
		"""
		Initialize the document store for the RAG system.
		"""
		document_store = MyPgVectorDocumentStore(
			embedding_dimension=768,
			vector_function="cosine_similarity",
			recreate_table=False,
			table_name="article_embeddings",
			connection_string=Secret.from_env_var("PG_CONN_STR"),
			sql_insert_string=INSERT_STATEMENT_STRING,
			sql_update_string=UPDATE_STATEMENT_STRING,
			language="french",
		)
		return document_store

	def init_document_embedder(self):
		"""Initialize the document embedder for the RAG system."""
		embedder_component = SentenceTransformersDocumentEmbedder(
			model=self.embedding_model_id,
			normalize_embeddings=True,
		)

		embedder_component.warm_up()
		return embedder_component

	def init_document_writer(self):
		"""Initialize the document writer for the RAG system."""
		document_writer = DocumentWriter(document_store=self.document_store)
		return document_writer

	def init_text_splitter(self):
		"""Initialize the text splitter for the RAG system."""
		text_splitter = RecursiveCharacterTextSplitterComponent(
			chunk_size=300, chunk_overlap=50
		)
		return text_splitter

	def init_document_cleaner(self):
		"initialize the component that will clean the document"
		document_cleaner = DocumentCleaner(
			remove_substrings=[r"This post has already been read \d+ times!"],
			remove_regex="",
			keep_id=True,
		)
		return document_cleaner

	def init_haystack_pipeline(self):
		"""Initialize the Haystack pipeline for the RAG system."""
		index_pipeline = Pipeline()
		index_pipeline.add_component("text_cleaner", self.document_cleaner)
		index_pipeline.add_component("text_splitter", self.text_splitter)
		index_pipeline.add_component("embedder", self.document_embedder)
		index_pipeline.add_component("writer", self.document_writer)

		index_pipeline.connect("text_cleaner", "text_splitter")
		index_pipeline.connect("text_splitter", "embedder")
		index_pipeline.connect("embedder", "writer")

		return index_pipeline

	def run(self, use_current_date: bool = True) -> None:
		"""
		Run the document processor pipeline.

		This method reads the documents from the database,
		clean the text, split the text into chunks,
		compute the embedding for each chunk and
		save the chunk as document in the document store.

		Args:
		    use_current_date (bool, optional): Whether to use the current date. Defaults to True.
		"""
		dataset = self.read_documents(
			table_name="article", use_current_date=use_current_date
		)
		haystack_documents = [
			Document(
				content=example.content,
				id=example.id,
				meta={
					"posted_at": example.posted_at,
					"url": example.url,
				},
			)
			for example in dataset
		]
		indexing_pipeline = self.init_haystack_pipeline()
		return indexing_pipeline.run(data={"documents": haystack_documents})
