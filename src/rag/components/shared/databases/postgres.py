import contextlib
import csv
from collections import defaultdict
from enum import Enum
from itertools import groupby
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

DISTANCE_OPS_MAPPING = {"cosine": "<=>", "l2": "<->", "inner": "<#>"}


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

	def create_embedding_index(
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
			"CREATE INDEX {if_not_exists} {index_name} ON {table} "
			" {config} ({column} vector_l2_ops)"
		).format(
			if_not_exists=sql.SQL("IF NOT EXISTS") if if_not_exists else sql.SQL(""),
			index_name=sql.Identifier(
				f"{self.namespace}_{table_name}_{column_name}_index"
			),
			table=self._full_table_name(table_name),
			column=sql.Identifier(column_name),
			config=sql.SQL(index_config),
		)
		with self._transaction() as cursor:
			cursor.execute(query)

	def create_full_text_index(self, table_name: str, column_name: str) -> None:
		"""
		Create a full-text search index on a specified column.

		Args:
		        table_name: Name of the table
		        column_name: Name of the column to index
		"""
		full_text_search_column = f"full_text_search_{column_name}"
		query = sql.SQL(
			"CREATE INDEX {index_name} ON {table} USING GIN ({tsvector_column})"
		).format(
			index_name=sql.Identifier(
				f"{self.namespace}_{table_name}_{full_text_search_column}_fts_index"
			),
			table=self._full_table_name(table_name),
			tsvector_column=sql.Identifier(full_text_search_column),
		)
		with self._transaction() as cursor:
			cursor.execute(query)
			logger.info(
				f"Full-text search index created on '{full_text_search_column}' in table '{table_name}'."
			)

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

		with self._transaction() as cursor:
			if returning:
				result = []
				for value in values:
					cursor.execute(
						sql.SQL(
							"{query} VALUES ({placeholders}) ON CONFLICT DO NOTHING RETURNING {returning}"
						).format(
							query=query,
							placeholders=sql.SQL(", ").join(
								sql.Placeholder() * len(value)
							),
							returning=sql.SQL(", ").join(
								map(sql.Identifier, returning)
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
		distance_op = DISTANCE_OPS_MAPPING[distance_metric]

		query = sql.SQL(
			"SELECT {similarity_expr} as similarity, {columns} FROM {table} ORDER BY similarity desc LIMIT {top_k}"
		).format(
			similarity_expr=self.distance_metrics_to_similarity_expression(
				distance_metric
			).format(
				vector_column=sql.Identifier(vector_column),
				op=sql.SQL(distance_op),
			),
			columns=sql.SQL(", ").join(map(sql.Identifier, return_columns)),
			table=self._full_table_name(table_name),
			vector_column=sql.Identifier(vector_column),
			op=sql.SQL(distance_op),
			top_k=sql.Literal(top_k),
		)
		with self._transaction() as cursor:
			if probe:
				cursor.execute(
					sql.SQL("SET LOCAL vchordrq.probes = %s").format(), (probe,)
				)
			cursor.execute(query, {"query_embedding": query_vector})
			return cursor.fetchall()

	def full_text_search(
		self,
		query_text: str,
		max_results: int,
		return_columns: List[str],
		column_name: str = "content",
		table_name: str = "documents",
	):
		"""
		Perform a full-text search on the specified table and column.
		Args:
		query_text: The search query string
		max_results: Maximum number of results to return
		column_name: The column to search (default: "content")
		table_name: The table to search (default: "documents")
		Returns:
		List of result rows
		"""
		full_text_search_column = f"full_text_search_{column_name}"
		query = sql.SQL(
			"""
		SELECT  ts_rank_cd({full_text_search_column}, websearch_to_tsquery(%(query_text)s)) as full_text_score, {columns}
		FROM {table}
		WHERE {full_text_search_column} @@ websearch_to_tsquery(%(query_text)s)
		ORDER BY full_text_score DESC
		LIMIT %(max_results)s
		"""
		).format(
			table=self._full_table_name(table_name),
			full_text_search_column=sql.Identifier(full_text_search_column),
			columns=sql.SQL(", ").join(map(sql.Identifier, return_columns)),
		)
		params = {"query_text": query_text, "max_results": max_results}
		with self._transaction() as cursor:
			cursor.execute(query, params)
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
				f"{self.namespace}_{table_name}_{column_name}_foreign_key"
			),
			column=sql.Identifier(column_name),
			foreign_table=self._full_table_name(foreign_table),
			foreign_column=sql.Identifier(foreign_column),
		)

		if if_not_exists:
			query = sql.SQL("ALTER TABLE IF NOT EXISTS {query}").format(query=query)

		with self._transaction() as cursor:
			cursor.execute(query)

	def create_database(self, database_name: str, if_not_exists: bool = True) -> None:
		"""
		Create a database if it does not exist.

		Args:
		    database_name: Name of the database to create
		    if_not_exists: Whether to add IF NOT EXISTS clause
		"""
		query = sql.SQL("CREATE DATABASE {if_not_exists} {database}").format(
			if_not_exists=sql.SQL("IF NOT EXISTS") if if_not_exists else sql.SQL(""),
			database=sql.Identifier(database_name),
		)

		with self._transaction() as cursor:
			cursor.execute(query)
			logger.info(f"Database '{database_name}' created or already exists.")

	def drop_constraint(
		self, table_name: str, constraint_name: str, if_exists: bool = True
	) -> None:
		"""
		Drop a constraint from a table.

		Args:
		    table_name: Name of the table
		    constraint_name: Name of the constraint to drop
		    if_exists: Whether to add IF EXISTS clause
		"""
		query = sql.SQL(
			"ALTER TABLE {table} DROP CONSTRAINT {if_exists} {constraint}"
		).format(
			table=self._full_table_name(table_name),
			if_exists=sql.SQL("IF EXISTS") if if_exists else sql.SQL(""),
			constraint=sql.Identifier(constraint_name),
		)

		with self._transaction() as cursor:
			cursor.execute(query)
			logger.info(f"Constraint '{constraint_name}' dropped from '{table_name}'.")

	def add_text_search_field(self, table_name: str, column_name: str):
		"""
		Add a tsvector column for full-text search.
		Args:
		    table_name: Name of the table
		    column_name: The column to generate the tsvector from
		"""
		# Create the full column name for the tsvector field
		full_text_search_column = f"full_text_search_{column_name}"

		query = sql.SQL(
			"ALTER TABLE {table} ADD COLUMN {tsvector_column} tsvector GENERATED ALWAYS AS (to_tsvector('english', {content_column})) STORED"
		).format(
			table=self._full_table_name(table_name),
			tsvector_column=sql.Identifier(full_text_search_column),
			content_column=sql.Identifier(column_name),
		)

		with self._transaction() as cursor:
			print(query.as_string(cursor))
			cursor.execute(query)
			logger.info(
				f"Column '{full_text_search_column}' added to '{table_name}' for full-text search."
			)

	def search_many_by_vector(
		self,
		table_name: str,
		vector_column: str,
		query_vectors: List[List[float]],
		return_columns: Sequence[str],
		distance_metric: DistanceMetric = DistanceMetric.COSINE,
		candidate_limit: int = 10,
		probe: Optional[int] = None,
	) -> List[Tuple[Any, ...]]:
		"""
		Perform multiple vector similarity searches using LATERAL JOIN for efficiency.
		Args:
		    table_name: Name of the table
		    vector_column: Name of the vector column
		    query_vectors: List of query vectors (n x m)
		    return_columns: Columns to return in results
		    distance_metric: One of "cosine", "l2", or "inner"
		    candidate_limit: Number of candidates to consider per query
		    probe: Number of probes for approximate search

		Returns:
		    List of result rows (one per query vector)
		"""
		distance_op = DISTANCE_OPS_MAPPING[distance_metric]

		if distance_metric == DistanceMetric.COSINE:
			# Cosine similarity (1 - distance)
			similarity_expr = "(1 - ({vector_column} {op} qv.query_vector::vector))"
		elif distance_metric == DistanceMetric.INNER_PRODUCT:
			# Inner product similarity: negate the pgvector operator result
			# to get the positive inner product value.
			similarity_expr = "(-({vector_column} {op} qv.query_vector::vector))"
		else:  # L2
			# L2 similarity: a common way to turn distance into similarity
			similarity_expr = "(1 / (1 + {vector_column} {op} qv.query_vector::vector))"

		array_constructor, params = self.convert_array_to_pg_vectors(query_vectors)

		query = sql.SQL(
			f"""
            SELECT
                results.*,
                qv.idx as query_index
            FROM
                unnest({array_constructor}) WITH ORDINALITY AS qv(query_vector, idx)
            CROSS JOIN LATERAL (
                SELECT
                    {{similarity}} as similarity,
                    {{columns}}
                FROM {{table}}
                ORDER BY similarity desc
                LIMIT {candidate_limit}
            ) AS results
            ORDER BY qv.idx, results.similarity DESC
        """
		).format(
			similarity=sql.SQL(
				similarity_expr.format(
					# Use quotes to prevent issues with reserved keywords
					vector_column='"{}"'.format(vector_column),
					op=distance_op,
				)
			),
			columns=sql.SQL(", ").join(map(sql.Identifier, return_columns)),
			return_columns=sql.SQL(", ").join(map(sql.Identifier, return_columns)),
			table=self._full_table_name(table_name),
			candidate_limit=sql.Literal(candidate_limit),
			vector_column=sql.Identifier(vector_column),
			op=sql.SQL(distance_op),
		)

		with self._transaction() as cursor:
			if probe:
				cursor.execute(sql.SQL("SET LOCAL vchordrq.probes = %s"), (probe,))
			print(query.as_string(cursor))
			cursor.execute(f"explain analyze {query.as_string(cursor)}", params)
			results = cursor.fetchall()

			return self._extract_top_results(results, k=candidate_limit)

	def _extract_top_results(
		self, results: List[Tuple], k: int = 1
	) -> List[Tuple[Any, ...]]:
		"""
		Groups results by query and returns the top k results for each.

		Args:
		    results: A list of result rows from the database, ordered by query index
		             and then by similarity. Each row is (similarity, column1, ..., query_index).
		    num_queries: The total number of query vectors.
		    k: The number of top results to return for each query.

		Returns:
		    A flattened list of the top k result rows for all queries.
		"""
		if not results:
			return []

		# Use a defaultdict to group results by their query index
		grouped_results = defaultdict(list)
		for index, group in groupby(results, key=lambda x: x[-1]):
			grouped_results[index] = list(group)[:k]
		return grouped_results

	def distance_metrics_to_similarity_expression(self, distance_metric: str) -> str:
		if distance_metric == DistanceMetric.COSINE:
			similarity_expr = sql.SQL(
				"(1 - ({vector_column} {op} %(query_embedding)s::vector))"
			)
		elif distance_metric == DistanceMetric.INNER_PRODUCT:
			similarity_expr = sql.SQL(
				"({vector_column} {op}%(query_embedding)s::vector)"
			)
		else:  # L2 - convert distance to similarity
			similarity_expr = sql.SQL(
				"(1 / (1 + {vector_column} {op} %(query_embedding)s::vector))"
			)
		return similarity_expr

	def convert_array_to_pg_vectors(
		self, query_vectors: List[List[float]]
	) -> Tuple[List[str], Dict[str, str]]:
		"""
		convert a list of list to PostgresQL array format

		It return the vector placeholder in this format:

		'        array_constructor = f"ARRAY[{', '.join(vector_placeholders)}]"
		and parameters like this:
		vector_1:vector: values
		vector_2:vector: values
		...
		vector_n:vector:    values
		"""

		vector_placeholders = []
		params = {}
		for i, vector in enumerate(query_vectors):
			param_name = f"vector_{i}::vector"

			vector_string = "[" + ",".join(str(float(x)) for x in vector) + "]"
			vector_placeholders.append(f"%({param_name})s")
			params[param_name] = vector_string
		array_constructor = f"ARRAY[{', '.join(vector_placeholders)}]"

		return array_constructor, params
