from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

class EmbeddingModel:
    def __init__(self, model_name=None, mode = "hf"):
        self.model_name = model_name 
        self.mode = mode
        if mode == "hf":
            self.embed_model_hf = HuggingFaceEmbedding(model_name=self.model_name)
        elif mode == "cohere":
            self.embed_model_cohere = CohereEmbedding(
                api_key=os.getenv("COHERE_API_KEY"),
                model_name="embed-english-v3.0",
                input_type="search_query",
            )

    def get_embedding_model(self):
        if self.mode == "hf":
            return self.embed_model_hf
        elif self.mode == "cohere":
            return self.embed_model_cohere




if __name__ == "__main__":
    

    embedding_model = EmbeddingModel(model_name="jinaai/jina-embeddings-v2-base-en")

    embed_model = embedding_model.get_embedding_model()
    
    embeddings = embed_model.get_text_embedding("Hello World!")
    
    print(embeddings[:5])
    
    cohere_embed_model = EmbeddingModel(mode="cohere").get_embedding_model()
    
    embeddings = cohere_embed_model.get_text_embedding("Hello World!")
    
    print(embeddings[:5])

    print("Embedding model initialized successfully!")