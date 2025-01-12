import os
from dotenv import load_dotenv
from llama_index.agent.introspective import IntrospectiveAgentWorker, SelfReflectionAgentWorker
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec

class IntrospectiveAgentManager:
    def __init__(self, groq_api_key: str, tavily_api_key: str, llm_model: str, embed_model_name: str):
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        self.llm_model = llm_model
        self.embed_model_name = embed_model_name

        os.environ["GROQ_API_KEY"] = self.groq_api_key
        load_dotenv()

    def create_introspective_agent(self, verbose: bool = True):
        tavily_tool = TavilyToolSpec(api_key=self.tavily_api_key)

        llm = Groq(
            model=self.llm_model,
            temperature=0.0,
            api_key=self.groq_api_key
        )

        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

        self_reflection_agent_worker = SelfReflectionAgentWorker.from_defaults(
            llm=llm,
            verbose=verbose
        )

        tool_list = tavily_tool.to_tool_list()
        main_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tool_list,
            llm=llm,
            verbose=verbose
        )

        introspective_worker_agent = IntrospectiveAgentWorker.from_defaults(
            reflective_agent_worker=self_reflection_agent_worker,
            main_agent_worker=main_agent_worker,
            verbose=verbose
        )

        return introspective_worker_agent.as_agent(verbose=verbose)

def main():
    groq_api_key = "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"
    tavily_api_key = "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"
    llm_model = "llama3-8b-8192"
    embed_model_name = "BAAI/bge-small-en-v1.5"

    agent_manager = IntrospectiveAgentManager(
        groq_api_key=groq_api_key,
        tavily_api_key=tavily_api_key,
        llm_model=llm_model,
        embed_model_name=embed_model_name
    )

    introspective_agent = agent_manager.create_introspective_agent(verbose=True)

    response = introspective_agent.chat("Who is Mukti?")
    print(str(response))

if __name__ == "__main__":
    main()
