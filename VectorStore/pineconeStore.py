from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os

class PineconeSearchEngine:
    def __init__(self, 
                 index_name="nyd-hackathon-patanjali-labse",
                 embedding_model_name="lingtrain/labse-sanskrit",
                 llm_model_name="llama3-8b-8192",
                 dimension=768):
        load_dotenv()
        self.index_name = index_name
        self.dimension = dimension
        self._setup_models(embedding_model_name, llm_model_name)
        self._initialize_pinecone()
        
    def _setup_models(self, embedding_model_name, llm_model_name):
        self.embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = self.embedding_model
        Settings.llm = Groq(model=llm_model_name, api_key=os.getenv("GROQ_API_KEY"))

    def _initialize_pinecone(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
        self.pinecone_index = self.pc.Index(self.index_name)

    def create_vector_store(self, data_path):
        documents = SimpleDirectoryReader(data_path).load_data()
        
        vector_store = PineconeVectorStore(
            pinecone_index=self.pinecone_index,
            add_sparse_vector=True
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )
        
        return self.index

    def query(self, query_text, mode="hybrid"):
        if not hasattr(self, 'index'):
            raise ValueError("Please create vector store first using create_vector_store()")
            
        query_engine = self.index.as_query_engine(vector_store_query_mode=mode)
        response = query_engine.query(query_text)
        return response

if __name__ == "__main__":
    engine = PineconeSearchEngine()
    engine.create_vector_store("./Data/Patanjali_Yoga_Sutras")
    
    response = engine.query("ततः क्लेशकर्मनिवृत्तिः")
    print(response)
