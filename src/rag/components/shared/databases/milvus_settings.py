from pydantic import BaseModel, Field


class MilvusSettings(BaseModel):
	uri: str = Field(
		"local_data/private_gpt/milvus/milvus_local.db",
		description="The URI of the Milvus instance. For example: 'local_data/private_gpt/milvus/milvus_local.db' for Milvus Lite.",
	)
	token: str = Field(
		"",
		description=(
			"A valid access token to access the specified Milvus instance. "
			"This can be used as a recommended alternative to setting user and password separately. "
		),
	)
	collection_name: str = Field(
		"make_this_parameterizable_per_api_call",
		description="The name of the collection in Milvus. Default is 'make_this_parameterizable_per_api_call'.",
	)
	vector_dimension: int = Field(
		1024,
		description=(
			"The dimension of the vector. "
			"This should match the dimension of the embeddings used in the collection."
		),
	)
