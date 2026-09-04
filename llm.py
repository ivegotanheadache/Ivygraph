from pydantic import BaseModel, Field
from typing import List, Literal, Type, TypeVar
from abc import ABC, abstractmethod
import logging
import json


from openai import OpenAI, AuthenticationError
from config import MAX_HYPONYMS, MAX_SYNONYMS, BACKEND, LOCAL_LLAMA_PATH

import os
from dotenv import load_dotenv
load_dotenv()
APIKEY = os.getenv("OPENAI_API_KEY")

class HypernymResponse(BaseModel):
    hypernym: str
    pos: Literal["NOUN", "VERB", "ADJ", "ADV"]

class HyponymsResponse(BaseModel):
    words: List[str] = Field(min_length=1, max_length=MAX_HYPONYMS)  

class SynonymsResponse(BaseModel):
    words: List[str] = Field(min_length=1, max_length=MAX_SYNONYMS)  

T = TypeVar("T", bound=BaseModel)

class BaseLLM(ABC):
    @abstractmethod
    def create_structured_completion(
        self,
        messages: list,
        schema: Type[T],
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> T:
        raise NotImplementedError



class OpenAILLM(BaseLLM):
    def __init__(self):
        if not APIKEY:
            logging.warning("OpenAILLM: nessuna API key fornita")
        self.client = OpenAI(api_key=APIKEY)
        try:
            self.client.models.list()
        except AuthenticationError as e:
            logging.error(f"OpenAILLM: API key non valida o revocata: {e}")
            raise
        except Exception as e:
            logging.warning(f"OpenAILLM: impossibile verificare la API key ora (probabile problema di rete): {e}")

    def create_structured_completion(self, messages, schema, max_tokens=100, temperature=0.7):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=schema,  
        )
        return response.choices[0].message.parsed



class LocalLlamaLLM(BaseLLM):
    
    def __init__(self, n_ctx=4096, n_threads=10, n_batch=128):
        try: 
            from llama_cpp import Llama, LlamaGrammar
        except ImportError as e:
            logging.error(f"LocalLlamaLLM: impossibile importare llama_cpp: {e}")
            raise

        if not LOCAL_LLAMA_PATH or not os.path.isfile(LOCAL_LLAMA_PATH):
            raise FileNotFoundError(
                f"LocalLlamaLLM: modello non trovato al path '{LOCAL_LLAMA_PATH}' "
                f"(controlla config.LOCAL_LLAMA_PATH)"
            )
        try:
            self.model = Llama(
                model_path=LOCAL_LLAMA_PATH,
                verbose=False,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
            )
        except Exception as e:
            logging.error(f"LocalLlamaLLM: impossibile caricare il modello da '{LOCAL_LLAMA_PATH}': {e}")
            raise

    def create_structured_completion(self, messages, schema, max_tokens=100, temperature=0.7):
        json_schema = json.dumps(schema.model_json_schema())
        grammar = LlamaGrammar.from_json_schema(json_schema)

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar,   
        )
        raw_json = response["choices"][0]["message"]["content"]
        return schema.model_validate_json(raw_json)

def get_llm() -> BaseLLM:
    match BACKEND:
        case "openai":
            return OpenAILLM()
        case "llama":
            return LocalLlamaLLM()
        case _:
            raise ValueError(f"Backend sconosciuto in config.BACKEND: '{BACKEND}' (atteso 'openai' o 'llama')")

llm = get_llm()