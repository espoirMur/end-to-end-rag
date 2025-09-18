import json
import os
from typing import Dict, List

import requests
from jinja2 import Template
from transformers import AutoTokenizer

from src.rag.schemas.generator_schemas import AnswerModel

RAG_PROMPT_TEMPLATE = """
            DOCUMENTS:
            {% for document in documents %}
             - [{{ loop.index }}] {{document}} \n
            {% endfor %}

            QUESTION:
           {{question}}
            INSTRUCTIONS:
            Answer the users QUESTION using the DOCUMENTS text above.
            Keep your answer ground in the facts of the DOCUMENTS.
            If you can't answer based on the context say that answers is correct = False
            When you found the question mention the document number it was found in
        """


class LLamaCppGeneratorComponent:
	"""
	This class is responsible for generating response using the Llamma.cpp api

	"""

	def __init__(
		self,
		api_url: str,
		prompt: str,
		model_name: str = "croissantllm/CroissantLLMChat-v0.1",
		json_schema: dict = AnswerModel.model_json_schema(),
		lll_model_name: str = "deepseek-chat",
	) -> None:
		self.api_url = api_url
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.prompt = prompt
		self.json_schema = json_schema
		self.load_api_key_from_env()
		self.lll_model_name = lll_model_name

	def load_api_key_from_env(self) -> None:
		"""Load the API key from the environment variables"""
		self.api_key = os.getenv("DEEP_SEEK_API_KEY")

	def generate_chat_input(
		self, template_values: dict, prompt_template: str = RAG_PROMPT_TEMPLATE
	) -> List[Dict]:
		"""generate the prompt to be used for the chat input"""

		template = Template(prompt_template)
		rendered_prompt = template.render(**template_values)

		chat_input = [
			{"role": "system", "content": self.prompt},
			{"role": "user", "content": rendered_prompt},
		]

		return chat_input

	def generate_response(self, prompt: str) -> str:
		"""
		This function generates response using the Llamma.cpp api

		Args:
		    prompt (str): The prompt to generate response from

		Returns:
		    str: The generated response
		"""
		headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {self.api_key}",
		}

		data = {
			"prompt": prompt,
			"n_predict": 768,
			"temperature": 0.3,
			"top_k": 40,
			"top_p": 0.90,
			"stopped_eos": True,
			"repeat_penalty": 1.05,
			"stop": ["assistant", self.tokenizer.eos_token],
			"seed": 42,
			"json_schema": self.json_schema,
		}

		json_data = json.dumps(data)

		# Send the POST request
		try:
			response = requests.post(
				f"{self.api_url}/completion",
				headers=headers,
				data=json_data,
				timeout=300,
			)
			return response.json()["content"]
		except requests.exceptions.RequestException as e:
			print(e)
			return None

	def run(
		self, template_values: dict, prompt_template: str = RAG_PROMPT_TEMPLATE
	) -> str:
		"""Generate response using the Llamma.cpp api"""
		chat_input = self.generate_chat_input(template_values, prompt_template)
		chat_tokens = self.tokenizer.apply_chat_template(
			chat_input, tokenize=False, add_generation_prompt=True
		)
		response = self.generate_response(chat_tokens)
		return response

	def _ping_api(self) -> bool:
		"""Ping the Llamma.cpp api to check if it is up"""
		try:
			response = requests.get(f"{self.api_url}/health", timeout=20)
			return response.status_code == 200 and response.json()["status"] == "ok"
		except requests.exceptions.RequestException as e:
			print(e)
			return False
