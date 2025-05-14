from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.api.schemas import Query
from src.api.services.retrieval import RetrieverServices

router = APIRouter()


@router.post("/retriever")
def retrieve(request: Request, query: Query) -> JSONResponse:
	"""retrieve the top k documents from the database given the query"""
	retriever_services = request.state.injector.get(RetrieverServices)
	retrieved_context = retriever_services.retrieve(query.query)
	retrieved_context_json = jsonable_encoder(retrieved_context)
	return JSONResponse(content=retrieved_context_json)
