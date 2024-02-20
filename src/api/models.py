from os import getenv
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, text, Text
from sqlalchemy.orm import mapped_column
from pgvector.sqlalchemy import Vector
from typing import List

load_dotenv()
database_user = getenv('POSTGRES_USER')
database_password = getenv('POSTGRES_PASSWORD')
database_host = getenv('POSTGRES_HOST')
database_port = getenv('POSTGRES_PORT')
database_name = getenv('POSTGRES_DB')

DATABASE_URL = f'postgresql+psycopg2://{database_user}:{quote(database_password)}@{database_host}:{database_port}/{database_name}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
Base = declarative_base()

VECTOR_LENGTH = 1024


class VectorModel(Base):
    __tablename__ = 'pubmed_qa'
    id: int = Column(Integer, primary_key=True)
    context: str = Column(Text)
    context_vector = mapped_column(Vector(VECTOR_LENGTH))

    def __len__(self) -> int:
        """conveniance method to get the length of the vector

        Returns:
            _type_: _description_
        """
        return VECTOR_LENGTH
