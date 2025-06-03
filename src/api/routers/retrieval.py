from typing import List

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.api.schemas import Query
from src.api.services.retrieval import RetrieverServices
from src.rag.schemas.document import Node

router = APIRouter()


@router.post("/retrieve", response_model=List[Node])
def retrieve(request: Request, query: Query) -> JSONResponse:
	"""retrieve the top k documents from the database given the query"""
	retriever_services: RetrieverServices = request.state.injector.get(
		RetrieverServices
	)
	retrieved_context = retriever_services.search(query.query)
	retrieved_context_json = jsonable_encoder(retrieved_context)
	return JSONResponse(content=retrieved_context_json, status_code=200)
