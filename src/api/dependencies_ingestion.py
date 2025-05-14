from injector import Injector

from src.api.schemas import ModelName
from src.rag.components.shared.databases.milvus_settings import MilvusSettings


def create_application_injector() -> Injector:
	_injector = Injector(auto_bind=True)

	# Need to replace this with the setting from environments
	COLLECTION_NAME = "my_collection"
	embedding_model_name = "intfloat/multilingual-e5-large"
	settings = MilvusSettings(
		uri="http://localhost:19530",
		collection_name=COLLECTION_NAME,
		vector_dimension=1024,
	)

	_injector.binder.bind(
		MilvusSettings,
		to=settings,
	)
	_injector.binder.bind(
		ModelName,
		to=ModelName(embedding_model_name),
	)
	return _injector


"""
Global injector for the application.

Avoid using this reference, it will make your code harder to test.

Instead, use the `request.state.injector` reference, which is bound to every request
"""
global_injector: Injector = create_application_injector()
