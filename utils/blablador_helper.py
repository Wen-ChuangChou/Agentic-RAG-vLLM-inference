import requests
import json
from openai import OpenAI
from typing import Optional

class Models:

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def get_model_data(self):
        response = requests.get(url=f"{self.base_url}/models",
                                headers=self.headers)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()["data"]

    def get_model_ids(self):
        model_data = self.get_model_data()
        return [model["id"] for model in model_data]


class ChatCompletions:

    def __init__(self,
                 api_key,
                 model,
                 temperature=0.7,
                 choices=1,
                 max_tokens=100,
                 user='default'):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def get_completion(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "user": self.user
        }

        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload  # Use json parameter instead of data + json.dumps
        )

        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()


class Completions:

    def __init__(self,
                 api_key,
                 model,
                 temperature=0.7,
                 choices=1,
                 max_tokens=50,
                 user="default"):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def get_completion(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "user": self.user
        }

        response = requests.post(url=f"{self.base_url}/completions",
                                 headers=self.headers,
                                 json=payload)

        response.raise_for_status()
        return response.json()


class TokenCount:

    def __init__(self, model, max_tokens=0):
        self.model = model
        self.max_tokens = max_tokens
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1/token_check"

    def count(self, prompts):
        try:
            iterator = iter(prompts)
        except TypeError:
            prompt_list = [{
                "model": self.model,
                "prompt": prompts,
                "max_tokens": self.max_tokens
            }]
        else:
            prompt_list = []
            for prompt in prompts:
                prompt_list.append({
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens
                })

        payload = {"prompts": prompt_list}

        response = requests.post(url=self.base_url,
                                 headers=self.headers,
                                 json=payload)

        response.raise_for_status()
        return response.json()


class BlabladorChatModel:
    """
    A class that initializes an API call to Helmholtz Blablador 
    using the official OpenAI client library.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=3
        )

    def get_model_data(self):
        response = requests.get(url=f"{self.base_url}/models",
                                headers=self.headers)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()["data"]

    def get_model_fullname(self, model: str) -> str:
        models = self.client.models.list()
        model_ids = [model.id for model in models.data]
        model_index = next((i for i, model_name in enumerate(model_ids) if model in model_name),None)
        model_fullname = model_ids[model_index]

        if model_index is not None:
            print(f"Model {model_fullname} found")              
        else:
            raise ValueError(f"Model {model} not found")
        return model_fullname  

    def get_response(self, prompt: str, model: str = "alias-fast") -> Optional[str]:
        """
        Send a prompt to a specified LLM on Blablador and return its response string.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str): The model to use for the LLM.

        Returns:
            Optional[str]: The response from the LLM.
        """

        model_fullname = self.get_model_fullname(model)

        try:
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")

            response = self.client.chat.completions.create(
                model=model_fullname,
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=500,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

