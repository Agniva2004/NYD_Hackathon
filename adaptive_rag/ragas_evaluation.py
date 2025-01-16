import pandas as pd
import random
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
import os

import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


set_llm_cache(GPTCache(init_gptcache))


os.environ["GROQ_API_KEY"] = "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"

def llm_eval(response, query, reference, documents, model="mixtral-8x7b-32768"):
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set in the .env file.")

    llm = ChatGroq(
        model=model,
        api_key=groq_api_key,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    evaluator_llm = LangchainLLMWrapper(llm)


    dataset = []

    dataset.append(
            {
                "user_input": query,
                "response": response,           
                "reference": reference,
                "retrieved_contexts": documents      
            }
        )
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness()],
        llm=evaluator_llm
    )

    return result