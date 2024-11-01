from fastapi import APIRouter
from fastapi.responses import JSONResponse
from numpy import array as np_array
from sqlalchemy import select

from api.models import VectorModel, session
from api.serializers import InputVector

router = APIRouter()


@router.post("/similar-vectors/")
async def get_similar_vectors(input_vector: InputVector):
	# Convert the input vector to a pgvector
	input_pgvector = np_array(input_vector.vector)
	similar_statement = VectorModel.context_vector.cosine_distance(input_pgvector)
	statement = select(VectorModel).order_by(similar_statement).limit(5)

	# Perform similarity search
	similar_contexts = session.scalars(statement).all()
	similar_contexts = [context.context for context in similar_contexts]
	return JSONResponse(content=similar_contexts)
