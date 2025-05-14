from fastapi import Depends, FastAPI, Request
from injector import Injector

from src.api.routers.retrieval import router as retrieval_router
from src.api.services.retrieval import RetrieverServices


def create_app(root_injector: Injector) -> FastAPI:
	"""
	The main factory application that will create the fastapi instance.
	"""

	async def bind_injector_to_request(request: Request):
		"""
		Bind the injector to the request.
		"""
		request.state.injector = root_injector

	app = FastAPI(dependencies=[Depends(bind_injector_to_request)])

	app.include_router(
		retrieval_router,
	)
	# Init all the services here
	root_injector.get(RetrieverServices)

	@app.get("/")
	async def root():
		return {"message": "Welcome to my retrieval app!"}

	return app
