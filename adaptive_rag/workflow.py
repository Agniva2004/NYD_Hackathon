
from pprint import pprint
from adaptive_rag_class import ADAPTIVE_RAG
from stategraph import GraphState
from ReactAgent.react_agent import React_Agent
from langgraph.graph import END, StateGraph, START
    

class Workflow:
    def __init__(self, model, embd_model, api_key, k, csv_path):


        self.adaptive_rag = ADAPTIVE_RAG(model, embd_model, api_key, k, csv_path)
        self.workflow = StateGraph(GraphState)
        

        self.workflow.add_node("web_search", self.adaptive_rag.web_search)  # web search
        self.workflow.add_node("retrieve", self.adaptive_rag.retrieve)  # retrieve
        self.workflow.add_node("grade_documents", self.adaptive_rag.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.adaptive_rag.generate)  # generate
        self.workflow.add_node("transform_query", self.adaptive_rag.transform_query)  # transform query
        self.workflow.add_node("abstraction", self.adaptive_rag.abstraction)  # abstraction
        # self.workflow.add_node("track_recursion", self.adaptive_rag.track_recursion_and_retrieve)  # track recursion
        # self.workflow.add_node("end_due_to_limit", self.adaptive_rag.end_due_to_limit)  # End node when recursion limit is reached
        self.workflow.add_node("introspective_agent_response", self.adaptive_rag.introspective_agent_response)

        self.workflow.add_conditional_edges(
            START,
            self.adaptive_rag.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("retrieve", "abstraction")
        self.workflow.add_edge("abstraction", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.adaptive_rag.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")

        self.workflow.add_conditional_edges(
            "generate",
            self.adaptive_rag.grade_generation_v_documents_and_question,
            {
                "not supported": "introspective_agent_response",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        self.workflow.add_edge("introspective_agent_response", END)
        self.app = self.workflow.compile()

    def build_workflow(self):
        pass

    def run_workflow(self, inputs):
        for output in self.app.stream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")
            pprint("\n---\n")

        # print(value["generation"])
        return {"generation": value["generation"], "extracted_info" : value["extractions"]}
