from src.api.models import VectorModel
from src.api.models import session
from src.api.serializers import InputVector
from fastapi.responses import JSONResponse
from fastapi import APIRouter

router = APIRouter()


@router.post("/similar-vectors/")
async def get_similar_vectors(input_vector: InputVector):
    # Convert the input vector to a pgvector
    input_pgvector = VectorModel(vector=input_vector.vector)

    # Perform similarity search
    similar_vectors = session.query(VectorModel).filter(
        VectorModel.vector.op('%%')(input_pgvector)).all()

    # Return the top 5 similar vectors
    top_5_similar_vectors = similar_vectors[:5]

    # Convert the vectors to lists and return as JSON
    top_5_similar_vectors_as_lists = [
        vector.vector for vector in top_5_similar_vectors]
    return JSONResponse(content=top_5_similar_vectors_as_lists)
