

import os
os.environ["TAVILY_API_KEY"] = "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"
from typing import List
import numpy as np
from rouge import Rouge
from bert_score import score as bert_score  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For BLEU score (`pip install nltk`)
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub



from pydantic import BaseModel, Field
class Abstractor:
    def __init__(self, model, api_key):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt = '''You are a knowledgeable assistant specializing in retrieving information from sacred texts like the Bhagavad Gita and Patanjali's Yoga Sutras. Your task is to extract and provide the most relevant chapter, verse, and Sanskrit slokas based on a user's query.

        Here are the key instructions for your task:

        1. You will access a database containing information in the following columns:
        - Chapter
        - Verse
        - Sanskrit Slokas
        - Translation
        - Question

        2. When a user provides a query, your goal is to:
        - Identify the most relevant documents that match the query.
        - Extract and return the associated chapter, verse, and Sanskrit slokas from the matched documents.
        - Ensure the output is clear and concise.

        3. Format your response as follows:
        Relevant Information: Chapter: [Chapter Number] Verse: [Verse Number] Sanskrit Slokas: [Sanskrit Text]
        4. If no relevant documents are found, respond with: 
        "No relevant chapter, verse, or Sanskrit slokas were found for the given query. Please try a different query."

        5. Be precise and avoid including irrelevant details. Your response should focus only on the chapter, verse, and Sanskrit slokas associated with the query.

        6. Use the knowledge from the database and ensure the results align with the user's input context.

        ''' 
        self.abstractor_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Given the retrieved documents content {content} extract the most relevant chapter, verse, and Sanskrit slokas from the database.",
                ),
            ]
        )
        self.abstractor = self.abstractor_prompt | self.llm | StrOutputParser()
    
    def abstract(self, content):
        abstracted_info=self.abstractor.invoke({"content": content})  
        return abstracted_info