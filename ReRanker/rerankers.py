import os
from dotenv import load_dotenv
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank

class Rerankers:
    def __init__(self):
        load_dotenv()
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-reranker-v2-m3")
        Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        
    def get_cohere_reranker(self, top_n=5):
        return CohereRerank(
            api_key=os.getenv("COHERE_API_KEY"),
            top_n=top_n
        )
    
    def get_colbert_reranker(self, top_n=5):
        return ColbertRerank(
            top_n=top_n,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True
        )
    
    def get_sentence_transformer_reranker(self, top_n=5):
        return SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=top_n
        )
