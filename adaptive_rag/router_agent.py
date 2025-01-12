from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    
    
class Router:
    def __init__(self, model, api_key):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt ="""You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to vedanta, yoga and patanjali.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        self.structered_llm_router = self.llm.with_structured_output(RouteQuery)
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )
        self.question_router = self.route_prompt | self.structered_llm_router

    def route_question(self, question):
        source = self.question_router.invoke({"question": question})
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO VECTORSTORE---")
            return "vectorstore"
        