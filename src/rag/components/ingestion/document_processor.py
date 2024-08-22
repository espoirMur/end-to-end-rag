from datetime import datetime
from shared.database import generate_database_connection, execute_query, postgres_uri
from datasets import Dataset, Value, Features
from haystack import Document
from haystack.components.preprocessors import DocumentCleaner
from shared.document_store import MyPgVectorDocumentStore
from ingestion.document_splitter import RecursiveCharacterTextSplitterComponent
from haystack.utils.auth import Secret
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline


import os
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
        self.database_connection = generate_database_connection()
        self.text_splitter = self.init_text_splitter()
        self.document_cleaner = self.init_document_cleaner()
        self.document_store = self.init_document_store()
        self.embedding_model_id = embedding_model_id
        self.document_embedder = self.init_document_embedder()
        self.document_writer = self.init_document_writer()

    def create_tables(self):
        execute_query(self.database_connection, TABLE_CREATION_STRING)

    def read_documents(self, table_name: str = 'article') -> Dataset:
        """
        Read the table containing the document in the database into a Huggingface Dataset object.
        """
        features = Features({
            'id': Value('int32'),
            'title': Value('string'),
            'content': Value('string'),
            'summary': Value('string'),
            'posted_at': Value('string'),
            'website_origin': Value('string'),
            'url': Value('string'),
            'author': Value('string'),
            'saved_at': Value('string'),
        })
        congo_news_dataset = Dataset.from_sql(
            table_name, con=self.database_connection, features=features)
        return congo_news_dataset

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
            language="french"
        )
        return document_store

    def init_document_embedder(self):
        """ Initialize the document embedder for the RAG system.
        """
        embedder_component = SentenceTransformersDocumentEmbedder(
            model=self,
            normalize_embeddings=True,

        )

        embedder_component.warm_up()
        return embedder_component

    def init_document_writer(self):
        """ Initialize the document writer for the RAG system.
        """
        document_writer = DocumentWriter(document_store=self.document_store)
        return document_writer

    def init_text_splitter(self):
        """ Initialize the text splitter for the RAG system.
        """
        text_splitter = RecursiveCharacterTextSplitterComponent(
            chunk_size=300, chunk_overlap=50)
        return text_splitter

    def init_document_cleaner(self):
        "initialize the component that will clean the document"
        document_cleaner = DocumentCleaner(remove_substrings=[
            r"This post has already been read \d+ times!"],
            remove_regex="",
            keep_id=True)
        return document_cleaner

    def init_haystack_pipeline(self):
        """Initialize the Haystack pipeline for the RAG system.
        """
        index_pipeline = Pipeline()
        index_pipeline.add_component("text_cleaner", self.document_cleaner)
        index_pipeline.add_component("text_splitter", self.text_splitter)
        index_pipeline.add_component("embedder", self.document_embedder)
        index_pipeline.add_component("writer", self.document_writer)

        index_pipeline.connect("text_cleaner", "text_splitter")
        index_pipeline.connect("text_splitter", "embedder")
        index_pipeline.connect("embedder", "writer")

        return index_pipeline

    def run(self):
        """Run the document processor Pipeline

        Args:
            documents (Dataset): _description_
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        # ned to read document for today only in the database.
        dataset = self.read_documents()
        haystack_documents = [
            Document(content=example['content'], id=example["id"], meta={}) for example in dataset
        ]
        indexing_pipeline = self.init_haystack_pipeline()
        return indexing_pipeline.run(haystack_documents)
