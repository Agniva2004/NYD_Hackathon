from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from router_agent import Router
from search_agent import Search
from answer_grader_agent import AnswerGrader
from hallucinator_agent import HallucinationGrader
from grader_agent import Grader
from question_rewriter import QuestionRewriter
from abstractor_agent import Abstractor
from pprint import pprint
from ReactAgent.react_agent import React_Agent
from langgraph.graph import END
import os
os.environ["TAVILY_API_KEY"] = "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"
class LoadDocuments:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_documents(self):
        import pandas as pd

        # Read the CSV file
        df = pd.read_csv(self.csv_path)

        # Initialize the lists for documents, questions, and translations
        documents = []
        questions = []
        translations = []

        for _, row in df.iterrows():
           
            content = (
                f"Chapter {row['chapter']}, Verse {row['verse']}:\n"
                f"Sanskrit_Slokas: {row['sanskrit']}\n"
                f"Translation: {row['translation']}\n"
                f"Question: {row['question']}\n"
                f"Augmented Response: {row['augmented_response']}"
            )
            documents.append(Document(page_content=content))

           
            row_questions = row['question'].split('?')  
            questions.extend(row_questions)

            # Add the corresponding translation for each question
            translations.extend([row['translation']] * len(row_questions))

        return documents, questions, translations
    
class MyEmbeddings(EmbeddingFunction):
    def __init__(self, model: str):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, output_value="token_embeddings")
        return [emb.mean(axis=0).tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query], output_value="token_embeddings")
        return embedding[0].mean(axis=0).tolist()
    
        
class ADAPTIVE_RAG:
    def __init__(self, model, embd_model,  api_key, k, csv_path):
        self.load_documents = LoadDocuments(csv_path)
        self.documents, self.questions, self.translations = self.load_documents.load_documents()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=0
        )
        self.doc_splits = self.text_splitter.split_documents(self.documents)
        self.embd = MyEmbeddings(embd_model)
        
        self.vectorstore = Chroma.from_documents(
            documents=self.doc_splits,
            collection_name="rag-chroma",
            embedding=self.embd
        )
        self.retriever = self.vectorstore.as_retriever()
        self.model=model
        self.api_key=api_key
        self.k=k
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.rag_chain = hub.pull("rlm/rag-prompt") | self.llm | StrOutputParser()
        self.recursion_limit = 7
        self.recursion_counter = 0
        agent_instance = React_Agent()

        base_data_path = "../Data"  
        agent_instance.setup_yoga_sutras_tool(base_folder_path=base_data_path)

        agent_instance.setup_memory()

        self.function_calling_agent = agent_instance.setup_agent()
    def retrieve(self, state):
        question = state["question"]
        documents = self.retriever.invoke(question)   
        return {"documents": documents, "question": question}
    def abstraction(self, state):
        content = state["documents"]
        abstractor = Abstractor(self.model, self.api_key)
        extractions = abstractor.abstract(content)
        print("extracted info: ", extractions)
        return {"documents": content, "question": state["question"], "extractions": extractions}
    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question,"extractions": state["extractions"], "generation": generation}
    def grade_documents(self, state):
        question = state["question"]
        print("quest")
        documents = state["documents"]
        filtered_docs = []
        for d in documents:
            grader=Grader(self.model, self.api_key)
            grade = grader.grade_documents(question, d.page_content)
            if grade == "yes":
                filtered_docs.append(d)
            else:
                continue
        return {"documents": filtered_docs, "question": question}
    def transform_query(self, state):
        question = state["question"]
        documents = state["documents"]
        questionRewriter = QuestionRewriter(self.model, self.api_key)
        better_question = questionRewriter.re_write_question(question)
        return {"documents": documents, "question": better_question}
    def web_search(self, state):
        question = state["question"]
        search=Search(self.k)
        docs = search.web_search(question)
        return {"documents": docs, "question": question}
    def route_question(self, state):
        question = state["question"]
        print("---ROUTE QUESTION---")
        router = Router(self.model, self.api_key)
        source = router.route_question(question)
        if source == "web_search":
            return "web_search"
        elif source == "vectorstore":
            return "vectorstore"
    def decide_to_generate(self, state):
        question = state["question"]
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"
    def grade_generation_v_documents_and_question(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        print("---CHECK HALLUCINATIONS---")
        hallucinationGrader=HallucinationGrader(self.model, self.api_key)
        grade = hallucinationGrader.grade_hallucinations(documents, generation)
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            answerGrader=AnswerGrader(self.model, self.api_key)
            grade = answerGrader.grade_answer(question, generation)
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        
    def react_agent_response(self, state):
        question = state["question"]
        response = self.function_calling_agent.chat(question)
        return {"generation": str(response)}
        
    def track_recursion_and_retrieve(self, state):
        """
        Track the number of recursions when going to 'transform_query'.
        End the workflow if the recursion limit is reached.
        """
        if self.recursion_counter < self.recursion_limit:
            self.recursion_counter += 1
            return "retrieve"
        else:
            return "end_due_to_limit"

    def end_due_to_limit(self, state):
        """
        End the workflow because the recursion limit was reached.
        """
        print("Recursion limit reached. Ending workflow.")
        return END    