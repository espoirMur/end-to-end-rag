from os import getenv
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Field, Text
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
Base = declarative_base()



class VectorModel(Base):
    __tablename__ = 'vectors'
    id: int = Column(Integer, primary_key=True)
    context: str = Column(Text)
    context_vector: List[float] = Field(default=None, sa_column=Column(Vector(1536)))
    
