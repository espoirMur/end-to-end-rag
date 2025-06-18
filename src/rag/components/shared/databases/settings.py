from typing import NamedTuple, Optional

from pydantic import BaseModel, Field, PositiveInt


class DistanceOp(NamedTuple):
	"""
	Structure for keeping details of distance pgvector's operations.
	"""

	function_name: str
	operator: str
	score_formula: str  # formula to calculate score based on distance


DISTANCE_OPS = {
	"cosine": DistanceOp("vector_cosine_ops", "<=>", "1 - distance"),
	"l2": DistanceOp("vector_l2_ops", "<->", "distance * -1"),
	"l1": DistanceOp("vector_l1_ops", "<+>", "distance * -1"),
	"ip": DistanceOp("vector_ip_ops", "<#>", "distance * -1"),
	"bit_hamming": DistanceOp("bit_hamming_ops", "<~>", "distance * -1"),
	"bit_jaccard": DistanceOp("bit_jaccard_ops", "<%>", "distance * -1"),
	"sparsevec_l2": DistanceOp("sparsevec_l2_ops", "<->", "distance * -1"),
	"halfvec_l2": DistanceOp("halfvec_l2_ops", "<->", "distance * -1"),
}


class HNSWConfig(BaseModel):
	# pgvector's recommended range
	m: PositiveInt = Field(default=16, ge=4, le=100)
	ef_construction: PositiveInt = Field(default=64, ge=10, le=1000)
	metric: str = Field(default="cosine", pattern="^(cosine|l2|ip)$")


class PgVectorSettings(BaseModel):
	"""Settings for the PostgreSQL vector database with pgvector extension."""

	uri: str = Field(
		description="PostgreSQL connection URI. Example: postgresql://user:password@localhost:5432/dbname",
	)
	collection_name: str = Field(
		default="default_collection",
		description="Name of the collection/table for vector storage",
		min_length=1,
		max_length=63,
	)
	vector_dimension: int = Field(
		default=1024,
		description="Dimension of the vector embeddings",
		gt=0,
		le=16000,  # pgvector's current max dimensions
	)
	table_name: str = Field(
		default="vectors",
		description="Name of the table in PostgreSQL",
		min_length=1,
		max_length=63,
	)
	vector_size: Optional[int] = Field(
		default=None,
		description="Size of the vector (in elements). Must be positive if provided",
		gt=0,
	)

	hnsw_config: HNSWConfig = Field(
		default_factory=HNSWConfig,
		description="Configuration for HNSW index parameters",
	)
	create_index: bool = Field(
		default=True,
		description="Whether to automatically create HNSW index on table creation",
	)
	index_name: Optional[str] = Field(
		default=None,
		description="Name of the index to create. Defaults to 'idx_{table_name}_hnsw'",
		min_length=1,
		max_length=63,
	)
