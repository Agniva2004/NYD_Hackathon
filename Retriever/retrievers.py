import warnings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
load_dotenv()

warnings.filterwarnings("ignore")

class FusionRetrieval:
    def __init__(self, splitter=None, chunk_size=256, vector_top_k=5, bm25_top_k=10, fusion_weights=[0.6, 0.4], fusion_top_k=10, mode="relative_score", num_queries=1, use_async=True, verbose=False):
        embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-reranker-v2-m3")
        Settings.embed_model = embedding_model
        Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        self.splitter = splitter or SentenceSplitter(chunk_size=chunk_size)
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.fusion_weights = fusion_weights
        self.fusion_top_k = fusion_top_k
        self.mode = mode
        self.num_queries = num_queries
        self.use_async = use_async
        self.verbose = verbose
        self.index = None
        self.retriever = None

    def build_index(self, documents):
        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[self.splitter], show_progress=True
        )

    def setup_relative_score_retriever(self):
        if not self.index:
            raise ValueError("Index has not been built. Call build_index() first.")
        
        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=self.bm25_top_k
        )
        self.retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=self.fusion_weights,
            similarity_top_k=self.fusion_top_k,
            num_queries=self.num_queries,
            mode=self.mode,
            use_async=self.use_async,
            verbose=self.verbose,
        )
    def setup_distribution_based_retriever(self):
        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=self.bm25_top_k
        )
        self.retriever =  QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=self.fusion_weights,
            similarity_top_k=self.fusion_top_k,
            num_queries=self.num_queries,
            mode="dist_based_score",
            use_async=self.use_async,
            verbose=self.verbose,
        )
    
    def setup_reciprocal_reranking_retriever(self):
        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=self.bm25_top_k
        )
        self.retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.fusion_top_k,
            num_queries=self.num_queries,
            mode="reciprocal_rerank",
            use_async=self.use_async,
            verbose=self.verbose,
        )
        

    def retrieve(self, query):

        if not self.retriever:
            raise ValueError("Retrievers have not been set up. Call setup_retrievers() first.")
        
        return self.retriever.retrieve(query)

if __name__ == "__main__":
    fusion = FusionRetrieval(chunk_size=256, fusion_weights=[0.6, 0.4], verbose=True)

    documents = SimpleDirectoryReader(r"..\Data\Patanjali_Yoga_Sutras").load_data()

    fusion.build_index(documents)
    fusion.setup_relative_score_retriever()

    results = fusion.retrieve("Time periods of past and future are inherent in the underlying foundation of the universe")
    for result in results:
        print(f"Score: {result.score:.2f} - {result.text[:100]}...\n-----")

