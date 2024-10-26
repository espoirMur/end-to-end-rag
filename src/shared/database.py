from psycopg2 import connect
from os import getenv
from dotenv import load_dotenv
from urllib.parse import quote
from typing import List, Any, Optional, Tuple, Dict
from unicodedata import normalize as unicode_normalize
from collections.abc import Generator
from sqlalchemy.engine import Connection

from psycopg2.extras import NamedTupleCursor

load_dotenv()

database_user = getenv('POSTGRES_USER')
database_password = getenv('POSTGRES_PASSWORD')
database_host = getenv('POSTGRES_HOST')
database_port = getenv('POSTGRES_PORT')
database_name = getenv('POSTGRES_DB')


postgres_uri = f'postgresql://{database_user}:{quote(database_password)}@{database_host}:{database_port}/{database_name}'

default_database_crendentials = {
    'user': database_user,
    'password': database_password,
    'host': database_host,
    'port': database_port,
    'database': database_name
}


def generate_database_connection(database_crendentials: Optional[Dict] = default_database_crendentials) -> Connection:
    database_connection = connect(
        **database_crendentials
    )
    return database_connection


def execute_query(database_connection, query, params=None) -> Generator[List[NamedTupleCursor]]:
    """
    Execute a database query using the provided database connection.

    Args:
        database_connection: A connection to the database.
        query: The SQL query to execute.
        params: Optional parameters to pass to the query.

    Returns:
        A list of query results, or None if an error occurs.
    """
    with database_connection.cursor(cursor_factory=NamedTupleCursor) as cursor:
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            raise ValueError(
                f"an execution error occurred for query {query!r} with params {params!r}") from e
