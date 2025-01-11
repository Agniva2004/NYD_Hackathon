import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from llama_index.core.callbacks import CallbackManager
from Data_Connector.data_connectors import DataConnector
from Chunking_TextSplitter.chunking_splitting import SemanticChunkerWrapper
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from Retriever.retrievers import FusionRetrieval
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.agent import AgentRunner
from llama_index.agent.coa import CoAAgentWorker
from llama_index.agent.llm_compiler import LLMCompilerAgentWorker
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.llm = Groq(model="llama3-8b-8192", temperature=0.0, api_key=os.getenv("GROQ_API_KEYS"))
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embedding_model

load_dotenv()

llm = Groq(model="llama3-8b-8192", temperature=0.0, api_key=os.getenv("GROQ_API_KEY"))
callback_manager = CallbackManager([])


base_folder_path = "../Data"
connector = DataConnector(base_folder_path=base_folder_path)

query_engine_tools = []

for subfolder in os.listdir(base_folder_path):
    subfolder_path = os.path.join(base_folder_path, subfolder)
    
    if os.path.isdir(subfolder_path) and subfolder == "Patanjali_Yoga_Sutras":
        documents = connector.fetch_files_with_llamaindex(subfolder)
        
        fusion = FusionRetrieval(chunk_size=256, fusion_weights=[0.6, 0.4], verbose=True)
        fusion.build_index(documents)
        fusion.setup_relative_score_retriever()
        
        query_engine = RetrieverQueryEngine.from_args(retriever=fusion.retriever)
        
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{subfolder.lower()}",
                    description=f"A query engine tool for the Patanjali Yoga Sutras, this tool is useful for questions related to the yoga sutras and the philosophy of yoga.",
                ),
            )
        )

for tool in query_engine_tools:
    print(f"Tool Name: {tool.metadata.name}, Description: {tool.metadata.description}")
    
worker = CoAAgentWorker.from_tools(
    tools=query_engine_tools,
    llm=Settings.llm,
    verbose=True,
)

agent = worker.as_agent()

answer = agent.chat("What is the meaning of life?")

print("COA Agent Response: \n")

print(str(answer))
