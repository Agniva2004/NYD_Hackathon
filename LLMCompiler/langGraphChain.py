from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Union, List, Literal, Dict
import ast
import os
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, BaseMessage
from joiner import _parse_joiner_output, select_recent_messages, JoinOutputs, FinalResponse, Replan
from plan_scheduler import plan_and_schedule
from langchain import hub
from langchain_groq import ChatGroq

class State(TypedDict):
    messages: list

class GraphChain:
    def __init__(self):
        self.graph_builder = StateGraph(State)
        self._configure_graph()
    
    def _configure_graph(self) -> None:
        """Configures the graph with nodes and edges."""
        self.graph_builder.add_node("plan_and_schedule", plan_and_schedule)
        joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
            examples=""
        )  
        llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

        runnable = joiner_prompt | llm.with_structured_output(JoinOutputs)
        self.graph_builder.add_node("join", runnable)
        
        self.graph_builder.add_edge(START, "plan_and_schedule")
        self.graph_builder.add_edge("plan_and_schedule", "join")
        
        self.graph_builder.add_conditional_edges(
            "join",
            self._should_continue,
            {
                "continue": "plan_and_schedule",  
                "end": END  
            }
        )
    
    def _should_continue(self, state: Dict) -> Union[Literal["continue"], Literal["end"]]:
        if "messages" not in state:
            return "continue"
        
        messages = state["messages"]
        if not messages:  
            return "continue"
        
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            return "end"
        return "continue"
    
    def return_chain(self):
        """Makes the class callable, delegating to the compiled chain."""
        chain = self.graph_builder.compile()
        return chain



