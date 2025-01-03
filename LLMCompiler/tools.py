import os
from typing import List, Tuple
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv

load_dotenv()

def create_yoga_sutras_tool() -> Tool:
    documents = CSVLoader(
        file_path="C:/Users/Srinjoy/OneDrive/Desktop/NYD/NYD_Hackathon/Data/Patanjali_Yoga_Sutras/Patanjali_Yoga_Sutras_Verses_English.csv",
        encoding="utf-8"
    ).load()
    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    template = """Given the following extracted chunks and a question, create a final answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

    question_generator = LLMChain(llm=llm, prompt=prompt)
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    
    retrieval_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
    )

    def search_sutras(query: str) -> str:
        chat_history: List[Tuple[str, str]] = []
        result = retrieval_chain.run({
            "question": query, 
            "chat_history": chat_history
        })
        return result

    return Tool(
        name="YogaSutrasSearch",
        func=search_sutras,
        description="Search and interpret the Yoga Sutras."
    )

def create_search_tool() -> Tool:
    return TavilySearchResults(
        max_results=1,
        description='tavily_search_results_json(query="the search query") - a search engine.',
    )
    

