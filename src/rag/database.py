from psycopg2 import connect
from os import getenv
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote
from typing import List, Any, Optional, Tuple
from unicodedata import normalize as unicode_normalize

from sqlalchemy.engine import Connection


load_dotenv()
database_user = getenv('POSTGRES_USER')
database_password = getenv('POSTGRES_PASSWORD')
database_host = getenv('POSTGRES_HOST')
database_port = getenv('POSTGRES_PORT')
database_name = getenv('POSTGRES_DB')


postgres_uri = f'postgresql://{database_user}:{quote(database_password)}@{database_host}:{database_port}/{database_name}'


def generate_database_connection() -> Connection:
    database_connection = connect(
        user=database_user,
        password=database_password,
        host=database_host,
        port=database_port,
        database=database_name
    )
    return database_connection


def execute_query(database_connection, query, params=None) -> Optional[List[Any]]:
    with database_connection.cursor() as cursor:
        cursor.execute(query, params)
        try:
            return cursor.fetchall()
        except:
            return None
