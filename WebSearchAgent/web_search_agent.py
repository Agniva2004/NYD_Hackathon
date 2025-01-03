import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec

class WebSearchAgent:
    def __init__(self, 
                 model_name: str = "llama3-8b-8192",
                 temperature: float = 0.0,
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 verbose: bool = True):
        """
        Initialize the WebSearchAgent with specified configurations.
        
        Args:
            model_name (str): Name of the Groq model to use
            temperature (float): Temperature setting for the LLM
            embedding_model_name (str): Name of the embedding model
            verbose (bool): Whether to enable verbose output
        """
        self._setup_environment()
        self.verbose = verbose
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model_name = embedding_model_name
        self.tools = []
        self.agent = None
        
        self._setup_models()
        self._setup_tools()

    def _setup_environment(self) -> None:
        """Setup environment variables and paths"""
        load_dotenv()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)

    def _setup_models(self) -> None:
        """Initialize LLM and embedding models"""
        self.llm = Groq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.llm = self.llm

        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name
        )
        Settings.embed_model = self.embedding_model

        self.callback_manager = CallbackManager([])

    def _setup_tools(self) -> None:
        tavily_tool = TavilyToolSpec(
            api_key=os.getenv("TAVILY_API_KEY"),
        )
        self.tools = tavily_tool.to_tool_list()

    def create_agent(self) -> FunctionCallingAgent:
        """
        Create and configure the function calling agent
        
        Returns:
            FunctionCallingAgent: Configured agent instance
        """
        self.agent = FunctionCallingAgent.from_tools(
            tools=self.tools,
            llm=Settings.llm,
            verbose=self.verbose
        )
        return self.agent

    def query(self, question: str) -> str:
        """
        Query the agent with a question
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The agent's response
        """
        if not self.agent:
            self.create_agent()
        
        try:
            response = self.agent.chat(question)
            return str(response)
        except Exception as e:
            return f"Error during query: {str(e)}"

    def reset_agent(self) -> None:
        """Reset the agent's state"""
        self.agent = None
        self._setup_tools()

def main():
    agent = WebSearchAgent(verbose=True)
    
    question = "What are the methods to achieve the goal of Yoga? What are the methods to quieten the fluctuations of the mind? What are the means to calm your thoughts? What is the importance of practice and dispassion?"
    print(f"\nQuestion: {question}")
    
    response = agent.query(question)
    print(f"Answer: {response}")

if __name__ == "__main__":
    main()
