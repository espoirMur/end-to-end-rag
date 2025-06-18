# copied from
# https://raw.githubusercontent.com/tensorchord/vechord/refs/heads/main/vechord/client.py
import contextlib
import contextvars
import math
from typing import Any, Optional, Sequence

import asyncpg
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from vechord.spec import (
	IndexColumn,
	Keyword,
	KeywordIndex,
	MultiVectorIndex,
	UniqueIndex,
	VectorIndex,
)

DEFAULT_TOKENIZER = ("bert_base_uncased", "wiki_tocken")

active_cursor = contextvars.ContextVar("active_cursor", default=None)
select_transaction_buffer = contextvars.ContextVar(
	"select_transaction_buffer", default=False
)


@contextlib.contextmanager
def limit_to_transaction_buffer():
	"""Only the rows inserted in the current transaction are returned."""
	token = select_transaction_buffer.set(True)
	try:
		yield
	finally:
		select_transaction_buffer.reset(token)


class VechordClient:
	"""A PostgreSQL client to access the database.

	Args:
	    namespace: used as a prefix for the table name.
	    url: the database connection URL.
	        e.g. "postgresql://user:password@localhost:5432/dbname"
	"""

	async def __init__(self, namespace: str, url: str):
		self.ns = namespace
		self.url = url
		self.connection = await asyncpg.connect(url, autocommit=True)
		register_vector(self.conn)

	async def create_extensions(
		self,
	):
		"""Create the transaction buffer and tokenizer extensions."""
		with self.connection.transaction() as connection:
			await connection.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
			await connection.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25")
			await connection.execute("CREATE EXTENSION IF NOT EXISTS pg_tokenizer")
			await connection.execute(
				'SET search_path TO "$user", public, bm25_catalog, tokenizer_catalog'
			)
			# may be register vector type here

	async def create_table_if_not_exists(
		self, name: str, schema: Sequence[tuple[str, str]]
	):
		columns = sql.SQL(", ").join(
			sql.SQL("{col} {typ}").format(
				col=sql.Identifier(col),
				typ=sql.SQL(typ.format(namespace=self.ns)),
			)
			for col, typ in schema
		)
		with self.connection.transaction() as connection:
			await connection.execute(
				sql.SQL("CREATE TABLE IF NOT EXISTS {table} ({columns});").format(
					table=sql.Identifier(f"{self.ns}_{name}"), columns=columns
				)
			)

	async def create_tokenizer(self):
		with self.connection.transaction() as connection:
			try:
				for name in DEFAULT_TOKENIZER:
					await connection(
						sql.SQL(
							"SELECT create_tokenizer({name}, $$model={model}$$)"
						).format(
							name=sql.Literal(name),
							model=sql.Identifier(name),
						)
					)
			except psycopg.errors.DatabaseError as err:
				if "already exists" not in str(err):
					raise

	async def create_index_if_not_exists(self, name: str, column: IndexColumn):
		with self.connection() as connection:
			if isinstance(column.index, UniqueIndex):
				query = sql.SQL(
					"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table} "
					"({column}) {config}"
				).format(
					index_name=sql.Identifier(self._index_name(name, column)),
					table=sql.Identifier(f"{self.ns}_{name}"),
					column=sql.Identifier(column.name),
					config=sql.SQL(column.index.config()),
				)
			else:
				query = sql.SQL(
					"CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
					"USING {index} ({column} {op_name})"
				).format(
					index_name=sql.Identifier(self._index_name(name, column)),
					table=sql.Identifier(f"{self.ns}_{name}"),
					index=sql.SQL(column.index.index),
					column=sql.Identifier(column.name),
					op_name=sql.SQL(column.index.op_name),
				)
				if config := column.index.config():
					query += sql.SQL(" WITH (options = $${config}$$)").format(
						config=sql.SQL(config)
					)
			await connection.execute(query)

	def _index_name(self, name: str, column: IndexColumn):
		return f"{self.ns}_{name}_{column.name}_{column.index.name}"

	def select(
		self,
		name: str,
		raw_columns: Sequence[str],
		kvs: Optional[dict[str, Any]] = None,
		from_buffer: bool = False,
		limit: Optional[int] = None,
	):
		"""Select from db table with optional key-value condition or from un-committed
		transaction buffer.

		- `from_buffer`: this ensures the select query only returns the rows that are
		    inserted in the current transaction.
		"""
		columns = sql.SQL(", ").join(map(sql.Identifier, raw_columns))
		cursor = self.get_cursor()
		query = sql.SQL("SELECT {columns} FROM {table}").format(
			columns=columns,
			table=sql.Identifier(f"{self.ns}_{name}"),
		)
		if kvs:
			condition = sql.SQL(" AND ").join(
				sql.SQL("{} IS NULL").format(sql.Identifier(col))
				if val is None
				else sql.SQL("{} = {}").format(
					sql.Identifier(col), sql.Placeholder(col)
				)
				for col, val in kvs.items()
			)
			query += sql.SQL(" WHERE {condition}").format(condition=condition)
		elif from_buffer:
			query += sql.SQL(" WHERE xmin = pg_current_xact_id()::xid;")
		if limit:
			query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
		# todo: handle by me
		cursor.execute(query, kvs)
		return [row for row in cursor.fetchall()]

	@staticmethod
	def _to_placeholder(kv: tuple[str, Any]):
		"""Process the `Keyword` type"""
		key, value = kv
		if isinstance(value, Keyword):
			return sql.SQL("tokenize({}, {})").format(
				sql.Placeholder(key), sql.Literal(value._model)
			)
		return sql.Placeholder(key)

	async def insert(self, name: str, values: dict):
		columns = sql.SQL(", ").join(map(sql.Identifier, values))
		placeholders = sql.SQL(", ").join(map(self._to_placeholder, values.items()))
		await self.connection.execute(
			sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders});").format(
				table=sql.Identifier(f"{self.ns}_{name}"),
				columns=columns,
				placeholders=placeholders,
			),
			values,
		)

	# todo: handle and check for better implementation
	async def copy_bulk(self, name: str, values: Sequence[dict], types: Sequence[str]):
		columns = sql.SQL(", ").join(map(sql.Identifier, values[0]))
		with self.transaction():
			cursor = self.get_cursor()
			with cursor.copy(
				sql.SQL(
					"COPY {table} ({columns}) FROM STDIN WITH (FORMAT BINARY)"
				).format(
					table=sql.Identifier(f"{self.ns}_{name}"),
					columns=columns,
				)
			) as copy:
				copy.set_types(types=types)
				for value in values:
					copy.write_row(tuple(value.values()))

	##

	async def delete(self, name: str, kvs: dict):
		if kvs:
			condition = sql.SQL(" AND ").join(
				sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder(col))
				for col in kvs
			)
			await self.connection.execute(
				sql.SQL("DELETE FROM {table} WHERE {condition};").format(
					table=sql.Identifier(f"{self.ns}_{name}"), condition=condition
				),
				kvs,
			)
		else:
			await self.connection.execute(
				sql.SQL("DELETE FROM {table};").format(
					table=sql.Identifier(f"{self.ns}_{name}")
				)
			)

	def query_vec(  # noqa: PLR0913
		self,
		name: str,
		vec_col: IndexColumn[VectorIndex],
		vec: np.ndarray,
		return_fields: Sequence[str],
		topk: int = 10,
		probe: Optional[int] = None,
	):
		columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
		if (
			probe is None
			and vec_col.index.lists is not None
			and vec_col.index.lists > 1
		):
			probe = math.ceil(vec_col.index.lists / 16)
		with self.transaction():
			cursor = self.get_cursor()
			cursor.execute(
				sql.SQL("SET LOCAL vchordrq.probes = {};").format(
					sql.Literal(probe or "")
				)
			)
			cursor.execute(
				sql.SQL(
					"SELECT {columns} FROM {table} ORDER BY {vec_col} {op} %s LIMIT %s;"
				).format(
					table=sql.Identifier(f"{self.ns}_{name}"),
					columns=columns,
					op=sql.SQL(vec_col.index.op_symbol),
					vec_col=sql.Identifier(vec_col.name),
				),
				(vec, topk),
			)
			return [row for row in cursor.fetchall()]

	def query_multivec(  # noqa: PLR0913
		self,
		name: str,
		multivec_col: IndexColumn[MultiVectorIndex],
		vec: np.ndarray,
		maxsim_refine: int,
		probe: Optional[int],
		return_fields: Sequence[str],
		topk: int = 10,
	):
		columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
		if (
			probe is None
			and multivec_col.index.lists is not None
			and multivec_col.index.lists > 1
		):
			probe = math.ceil(multivec_col.index.lists / 16)
		with self.transaction():
			cursor = self.get_cursor()
			cursor.execute(
				sql.SQL("SET LOCAL vchordrq.probes = {};").format(
					sql.Literal(probe or "")
				)
			)
			cursor.execute(
				sql.SQL("SET LOCAL vchordrq.maxsim_refine = {};").format(
					sql.Literal(maxsim_refine)
				)
			)
			cursor.execute(
				sql.SQL(
					"SELECT {columns} FROM {table} ORDER BY {multivec_col} @# %s LIMIT %s;"
				).format(
					table=sql.Identifier(f"{self.ns}_{name}"),
					columns=columns,
					multivec_col=sql.Identifier(multivec_col.name),
				),
				(vec, topk),
			)
			return [row for row in cursor.fetchall()]

	def query_keyword(  # noqa: PLR0913
		self,
		name: str,
		keyword_col: IndexColumn[KeywordIndex],
		keyword: str,
		return_fields: Sequence[str],
		tokenizer: str,
		topk: int = 10,
	):
		columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
		cursor = self.conn.execute(
			sql.SQL(
				"SELECT {columns} FROM {table} ORDER BY {keyword_col} <&> "
				"to_bm25query({index}, tokenize(%s, {tokenizer})) LIMIT %s;"
			).format(
				table=sql.Identifier(f"{self.ns}_{name}"),
				columns=columns,
				index=sql.Literal(self._index_name(name, keyword_col)),
				tokenizer=sql.Literal(tokenizer),
				keyword_col=sql.Identifier(keyword_col.name),
			),
			(keyword, topk),
		)
		return [row for row in cursor.fetchall()]

	def drop(self, name: str):
		self.conn.execute(
			sql.SQL("DROP TABLE IF EXISTS {table} CASCADE;").format(
				table=sql.Identifier(f"{self.ns}_{name}")
			)
		)
