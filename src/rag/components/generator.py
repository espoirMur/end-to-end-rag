import requests
import json
from transformers import AutoTokenizer
from typing import List, Dict
from jinja2 import Template


RAG_PROMPT_TEMPLATE = """
            DOCUMENTS:
            {% for document in documents %}
             - {{document}} \n
            {% endfor %}

            QUESTION:
           {{question}}
            INSTRUCTIONS:
            Answer the users QUESTION using the DOCUMENTS text above.
            Keep your answer ground in the facts of the DOCUMENTS.
            If the DOCUMENTS doesn’t contain the facts to answer the QUESTION return None
        """


class LLamaCppGeneratorComponent:
    """
    This class is responsible for generating response using the Llamma.cpp api

    """

    def __init__(self, api_url: str, prompt: str, model_name: str = "croissantllm/CroissantLLMChat-v0.1") -> None:
        self.api_url = api_url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = prompt

    def generate_chat_input(self, template_values: dict, prompt_template: str = RAG_PROMPT_TEMPLATE) -> List[Dict]:
        """ generate the prompt to be used for the chat input"""

        template = Template(prompt_template)
        prompt = template.render(**template_values)

        chat_input = [
            {"role": "system",
                "content": self.prompt},
            {"role": "user", "content": prompt},
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
            'Content-Type': 'application/json',
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
            "seed": 42
        }

        json_data = json.dumps(data)

        # Send the POST request
        try:
            response = requests.post(
                f"{self.api_url}/completion", headers=headers, data=json_data, timeout=300)
            return response.json()["content"]
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def run(self, template_values: dict, prompt_template: str = RAG_PROMPT_TEMPLATE) -> str:
        """Generate response using the Llamma.cpp api"""
        chat_input = self.generate_chat_input(
            template_values, prompt_template)
        chat_tokens = self.tokenizer.apply_chat_template(
            chat_input, tokenize=False, add_generation_prompt=True)
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
