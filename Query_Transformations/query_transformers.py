from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from dotenv import load_dotenv
import os

load_dotenv()

class QueryTransformer:
    def __init__(self, model_name="llama3-70b-8192"):
        self.llm = Groq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
        Settings.llm = self.llm
        self.query_gen_prompt = PromptTemplate("""\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate {num_queries} search queries, one on each line, \
related to the following input query:
Query: {query}
Queries:
""")
        self.hyde = HyDEQueryTransform(include_original=True)

    def generate_multiple_queries(self, query: str, num_queries: int = 4):
        response = self.llm.predict(
            self.query_gen_prompt,
            num_queries=num_queries,
            query=query
        )
        queries = response.split("\n")
        queries_str = "\n".join(queries)
        print(f"Generated queries:\n{queries_str}")
        return queries

    def generate_hypothetical_answer(self, query: str):
        query_bundle = self.hyde.run(query)
        return query_bundle.custom_embedding_strs

if __name__ == "__main__":
    transformer = QueryTransformer()
    
    queries = transformer.generate_multiple_queries("What is meditation?")
    
    hyde_result = transformer.generate_hypothetical_answer("What is karma?")
    print("\nHyDE Generated Answer:", hyde_result)



