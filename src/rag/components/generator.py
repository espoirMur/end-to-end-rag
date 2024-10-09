import requests
import json
from transformers import AutoTokenizer
from typing import List, Dict
from jinja2 import Template


class LLamaCppGeneratorComponent:
    """
    This class is responsible for generating response using the Llamma.cpp api

    """

    def __init__(self, api_url: str, prompt: str, model_name: str = "croissantllm/CroissantLLMChat-v0.1") -> None:
        self.api_url = api_url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = prompt

    def generate_chat_input(self, query: str, documents: list) -> List[Dict]:
        """ generate the prompt to be used for the chat input"""

        prompt_template = """
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
        template = Template(prompt_template)
        prompt = template.render(documents=documents, question=query)

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
            "n_predict": 512,
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
                f"{self.api_url}/completion", headers=headers, data=json_data, timeout=120)
            return response.json()["content"]
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def run(self, query: str, documents: list) -> str:
        """Generate response using the Llamma.cpp api"""
        chat_input = self.generate_chat_input(query, documents)
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
