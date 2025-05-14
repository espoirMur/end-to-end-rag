from src.api.app_factory import create_app
from src.api.dependencies_ingestion import global_injector

app = create_app(global_injector)
