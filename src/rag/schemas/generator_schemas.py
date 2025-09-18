from pydantic import BaseModel
from pydantic.v1.schema import schema


class Base(BaseModel):
	def to_json_schema(self) -> dict:
		model_schema = schema([self.__class__], ref_template="#/definitions/{model}")
		return model_schema


class AnswerModel(Base):
	text: str
	is_correct: bool


class QuestionListModel(Base):
	questions: list[str]
