import contextlib
import csv
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection, sql
from psycopg.pq import TransactionStatus

from src.shared.logger import setup_logger

logger = setup_logger("postgres database client")

# Type variables for better type hints
T = TypeVar("T")
Vector = np.ndarray
DEFAULT_TOKENIZER = "bert_base_uncased"


class DistanceMetric(str, Enum):
	"""Supported distance metrics for vector searches."""

	COSINE = "cosine"
	L2 = "l2"
	INNER_PRODUCT = "inner"


class PostgresVectorDBClient:
	"""An improved PostgreSQL client with vector search capabilities."""

	def __init__(
		self,
		namespace: str,
		connection: Connection,
		search_path: Sequence[str] = (
			"$user",
			"public",
			"bm25_catalog",
			"tokenizer_catalog",
		),
	):
		"""
		Initialize the VectorDB client.

		Args:
		    namespace: Prefix for table names
		    connection: Active PostgreSQL connection
		    search_path: PostgreSQL search path for extensions
		"""
		self.namespace = namespace
		self.connection = connection
		self._initialize_extensions(search_path)
		register_vector(self.connection)

	def _initialize_extensions(self, search_path: Sequence[str]):
		"""Initialize required PostgreSQL extensions."""
		with self._transaction() as cursor:
			cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
			cursor.execute(
				sql.SQL("SET search_path TO {}").format(
					sql.SQL(", ").join(map(sql.Identifier, search_path))
				)
			)

	@contextlib.contextmanager
	def _transaction(self):
		"""Context manager for database transactions."""
		if self.connection.info.transaction_status != TransactionStatus.IDLE:
			yield self.connection.cursor()
			return

		with self.connection.transaction():
			yield self.connection.cursor()

	def _full_table_name(self, name: str) -> sql.Identifier:
		"""Get the fully qualified table name with namespace."""
		return sql.Identifier(f"{self.namespace}_{name}")

	def create_table(self, name: str, schema: Dict, if_not_exists: bool = True) -> None:
		"""
		Create a table with the specified schema.

		Args:
		    name: Table name (without namespace prefix)
		    schema: Sequence of (column_name, type_definition) tuples
		    if_not_exists: Whether to add IF NOT EXISTS clause
		"""
		columns = sql.SQL(", ").join(
			sql.SQL("{column} {data_type}").format(
				column=sql.Identifier(column),
				data_type=sql.SQL(data_type),
			)
			for column, data_type in schema.items()
		)

		query = sql.SQL("CREATE TABLE {if_not_exists} {table} ({columns})").format(
			if_not_exists=sql.SQL("IF NOT EXISTS") if if_not_exists else sql.SQL(""),
			table=self._full_table_name(name),
			columns=columns,
		)

		with self._transaction() as cursor:
			cursor.execute(query)
			logger.info(f"Table {name} created with schema: {schema}")

	def create_index(
		self,
		table_name: str,
		column_name: str,
		index_config: str,
		unique: bool = True,
		if_not_exists: bool = True,
	) -> None:
		"""
		Create an index on a table column.

		Args:
		    table_name: Name of the table
		    column_name: Name of the column to index
		    index_config: Index configuration (e.g., "USING vchordrq")
		    unique: Whether to create a unique index
		    if_not_exists: Whether to add IF NOT EXISTS clause
		"""
		query = sql.SQL(
			"CREATE {unique} INDEX {if_not_exists} {index_name} ON {table} "
			"({column}) {config}"
		).format(
			unique=sql.SQL("UNIQUE") if unique else sql.SQL(""),
			if_not_exists=sql.SQL("IF NOT EXISTS") if if_not_exists else sql.SQL(""),
			index_name=sql.Identifier(
				f"{self.namespace}_{table_name}_{column_name}_idx"
			),
			table=self._full_table_name(table_name),
			column=sql.Identifier(column_name),
			config=sql.SQL(index_config),
		)

		with self._transaction() as cursor:
			cursor.execute(query)

	def insert(
		self,
		table_name: str,
		data: Dict[str, Any],
		returning: Optional[Sequence[str]] = None,
	) -> Optional[List[Tuple[Any, ...]]]:
		"""
		Insert a single record into the table.

		Args:
		    table_name: Name of the table
		    data: Dictionary of column names and values
		    returning: Optional columns to return after insert

		Returns:
		    List of returned rows if 'returning' specified, else None
		"""
		columns = sql.SQL(", ").join(map(sql.Identifier, data.keys()))
		placeholders = sql.SQL(", ").join(sql.Placeholder() * len(data))

		query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
			table=self._full_table_name(table_name),
			columns=columns,
			values=placeholders,
		)

		if returning:
			query = sql.SQL("{query} RETURNING {returning}").format(
				query=query,
				returning=sql.SQL(", ").join(map(sql.Identifier, returning)),
			)

		with self._transaction() as cursor:
			cursor.execute(query, list(data.values()))
			return cursor.fetchall() if returning else None

	def find_by_id_or_create(
		self, table_name: str, data: Dict[str, Any], id_field: str
	) -> Tuple[bool, Optional[List[Tuple[Any, ...]]]]:
		"""
		Find a record by primary key or create it if not found.

		Args:
		    table_name: Name of the table
		    data: Dictionary of column names and values

		Returns:
		    Tuple of (created, returned_rows)
		    created: True if a new record was created, False if found
		    returned_rows: Rows returned from the database
		"""
		columns = list(data.keys())
		values = list(data.values())

		# Check if the record exists
		find_query = sql.SQL(
			"SELECT * FROM {table} WHERE {id_field} = {id_value}"
		).format(
			table=self._full_table_name(table_name),
			id_field=sql.Identifier(id_field),
			id_value=sql.Placeholder(),  # Use the value from data
		)
		logger.info(f"Executing query: {find_query.as_string(self.connection)}")

		with self._transaction() as cursor:
			cursor.execute(find_query, [data.get(id_field, None)])
			rows = cursor.fetchall()

			if rows:
				return False, rows
			else:
				# Record not found, insert it
				insert_query = sql.SQL(
					"INSERT INTO {table} ({columns}) VALUES ({values})"
				).format(
					table=self._full_table_name(table_name),
					columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
					values=sql.SQL(", ").join(sql.Placeholder() * len(values)),
				)
				print(f"The insert query is: {insert_query.as_string(self.connection)}")
				cursor.execute(insert_query, values)
				return True, None

	def bulk_insert(
		self,
		table_name: str,
		data: Sequence[Dict[str, Any]],
		returning: Optional[Sequence[str]] = None,
	) -> Optional[List[Tuple[Any, ...]]]:
		"""
		Insert multiple records efficiently.

		Args:
		    table_name: Name of the table
		    data: Sequence of dictionaries with column names and values
		    returning: Optional columns to return after insert

		Returns:
		    List of returned rows if 'returning' specified, else None
		"""
		if not data:
			return None

		columns = list(data[0].keys())
		values = [tuple(item[col] for col in columns) for item in data]

		query = sql.SQL("INSERT INTO {table} ({columns})").format(
			table=self._full_table_name(table_name),
			columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
		)

		if returning:
			query = sql.SQL("{query} RETURNING {returning}").format(
				query=query,
				returning=sql.SQL(", ").join(map(sql.Identifier, returning)),
			)

		with self._transaction() as cursor:
			if returning:
				result = []
				for value in values:
					cursor.execute(
						sql.SQL("{query} VALUES ({placeholders})").format(
							query=query,
							placeholders=sql.SQL(", ").join(
								sql.Placeholder() * len(value)
							),
						),
						value,
					)
					result.extend(cursor.fetchall())
				return result
			else:
				formatted_query = sql.SQL(
					"{query}  values ({placeholders})  ON CONFLICT DO NOTHING"
				).format(
					query=query,
					placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
				)
				logger.info(
					f"Executing bulk insert query: {formatted_query.as_string(self.connection)}"
				)
				cursor.executemany(
					formatted_query,
					values,
				)
				return None

	def search_by_vector(
		self,
		table_name: str,
		vector_column: str,
		query_vector: Vector,
		return_columns: Sequence[str],
		top_k: int = 10,
		distance_metric: DistanceMetric = DistanceMetric.COSINE,
		probe: Optional[int] = None,
	) -> List[Tuple[Any, ...]]:
		"""
		Perform a vector similarity search.

		Args:
		    table_name: Name of the table
		    vector_column: Name of the vector column
		    query_vector: Query vector for similarity search
		    return_columns: Columns to return in results
		    top_k: Number of results to return
		    distance_metric: One of "cosine", "l2", or "inner"
		    probe: Number of probes for approximate search

		Returns:
		    List of result rows
		"""
		distance_op = {"cosine": "<=>", "l2": "<->", "inner": "<#>"}[distance_metric]

		query = sql.SQL(
			"SELECT {columns} FROM {table} ORDER BY {vector_column} {op} %s LIMIT %s"
		).format(
			columns=sql.SQL(", ").join(map(sql.Identifier, return_columns)),
			table=self._full_table_name(table_name),
			vector_column=sql.Identifier(vector_column),
			op=sql.SQL(distance_op),
		)

		with self._transaction() as cursor:
			if probe:
				cursor.execute(
					sql.SQL("SET LOCAL vchordrq.probes = %s").format(), (probe,)
				)
			cursor.execute(query, (query_vector, top_k))
			return cursor.fetchall()

	def search_by_keyword(
		self,
		table_name: str,
		text_column: str,
		query_text: str,
		return_columns: Sequence[str],
		tokenizer: str = DEFAULT_TOKENIZER,
		top_k: int = 10,
	) -> List[Tuple[Any, ...]]:
		"""
		Perform a full-text search using BM25.

		Args:
		    table_name: Name of the table
		    text_column: Name of the text column to search
		    query_text: Text query
		    return_columns: Columns to return in results
		    tokenizer: Tokenizer to use
		    top_k: Number of results to return

		Returns:
		    List of result rows
		"""
		query = sql.SQL(
			"SELECT {columns} FROM {table} ORDER BY {text_column} <&> "
			"to_bm25query(%s, tokenize(%s, %s)) LIMIT %s"
		).format(
			columns=sql.SQL(", ").join(map(sql.Identifier, return_columns)),
			table=self._full_table_name(table_name),
			text_column=sql.Identifier(text_column),
		)

		with self._transaction() as cursor:
			cursor.execute(query, (text_column, query_text, tokenizer, top_k))
			return cursor.fetchall()

	def close(self) -> None:
		"""Close the database connection."""
		self.connection.close()

	def insert_from_csv(
		self, table_name: str, csv_file: Path, encoding: str = "utf-8"
	) -> None:
		"""
		Insert data from a CSV file into the specified table.

		Args:
		    table_name: Name of the table
		    csv_file: Path to the CSV file
		    delimiter: Delimiter used in the CSV file
		    quotechar: Quote character used in the CSV file
		    encoding: Encoding of the CSV file
		"""
		columns = self.get_column_names_from_csv_header(Path(csv_file))
		insert_query = sql.SQL("COPY {table} ({columns}) FROM STDIN WITH csv").format(
			table=self._full_table_name(table_name),
			columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
		)
		with self._transaction() as cursor:
			with open(csv_file, "r", encoding=encoding) as f:
				cursor.copy_expert(
					insert_query,
					f,
				)

	def get_column_names_from_csv_header(self, csv_file_path: Path) -> List[str]:
		"""
		Extract column names from the header of a CSV file.

		Args:
		    csv_file_path: Path to the CSV file

		Returns:
		    List of column names
		"""
		with open(csv_file_path, "r", encoding="utf-8") as csvfile:
			dict_reader = csv.DictReader(csvfile)
			headers = dict_reader.fieldnames
			return list(headers)

	def add_foreign_key_to_table(
		self,
		table_name: str,
		column_name: str,
		foreign_table: str,
		foreign_column: str,
		if_not_exists: bool = True,
	) -> None:
		"""
		Add a foreign key constraint to a table.

		Args:
		    table_name: Name of the table to modify
		    column_name: Column to add the foreign key to
		    foreign_table: Referenced table
		    foreign_column: Referenced column in the foreign table
		    if_not_exists: Whether to add IF NOT EXISTS clause
		"""
		query = sql.SQL(
			"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
			"FOREIGN KEY ({column}) REFERENCES {foreign_table}({foreign_column})"
		).format(
			table=self._full_table_name(table_name),
			constraint=sql.Identifier(
				f"{self.namespace}_{table_name}_{column_name}_fk"
			),
			column=sql.Identifier(column_name),
			foreign_table=sql.Identifier(foreign_table),
			foreign_column=sql.Identifier(foreign_column),
		)

		if if_not_exists:
			query = sql.SQL("ALTER TABLE IF NOT EXISTS {query}").format(query=query)

		with self._transaction() as cursor:
			cursor.execute(query)
