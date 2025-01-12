import os
import sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Data_Connector.data_connectors import DataConnector
from Retriever.retrievers import FusionRetrieval

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.agent import AgentRunner
from llama_index.agent.coa import CoAAgentWorker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.llms.groq import Groq
from ReRanker.rerankers import Rerankers

class RetrievalAgent:
    def __init__(self):
        self.query_engine_tools = []
        self._setup_models()
        self.reranker = Rerankers()
        self.colbert_reranker = self.reranker.get_colbert_reranker()
        

    def _setup_models(self) -> None:
        """Initialize LLM and embedding models"""
        Settings.llm = Groq(
            model="llama3-8b-8192", 
            temperature=0.0, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def setup_yoga_sutras_tool(self, base_folder_path: str) -> None:
        connector = DataConnector(base_folder_path=base_folder_path)
        
        for subfolder in os.listdir(base_folder_path):
            subfolder_path = os.path.join(base_folder_path, subfolder)
            
            if os.path.isdir(subfolder_path):
                documents = connector.fetch_files_with_llamaindex(subfolder)
                
                fusion = FusionRetrieval(
                    chunk_size=256, 
                    fusion_weights=[0.6, 0.4], 
                    verbose=True
                )
                fusion.build_index(documents)
                fusion.setup_relative_score_retriever()
                
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=fusion.retriever
                )
                
                self.query_engine_tools.append(
                    QueryEngineTool(
                        query_engine=query_engine,
                        metadata=ToolMetadata(
                            name=f"vector_tool_{subfolder.lower()}",
                            description="A query engine tool for the Patanjali Yoga Sutras, useful for questions related to the yoga sutras and the philosophy of yoga.",
                        ),
                    )
                )


    def print_tools(self) -> None:
        for tool in self.query_engine_tools:
            if hasattr(tool, 'metadata') and tool.metadata.name is not None:
                print(f"Tool Name: {tool.metadata.name}")
                print(f"Description: {tool.metadata.description}\n")

    def setup_agent(self) -> CoAAgentWorker:
        return CoAAgentWorker.from_tools(
            tools=self.query_engine_tools,
            llm=Settings.llm,
            verbose=True
        )

def main():
    agent_retrieval = RetrievalAgent()
    
    agent_retrieval.setup_yoga_sutras_tool("../Data")
    
    agent_retrieval.print_tools()
    
    worker = agent_retrieval.setup_agent()
    agent = worker.as_agent()
    
    answer = agent.chat("What are the methods to achieve the goal of Yoga? What are the methods to quieten the fluctuations of the mind? What are the means to calm your thoughts? What is the importance of practice and dispassion?")
    print("\nAgent Response:")
    print(str(answer))

if __name__ == "__main__":
    main()