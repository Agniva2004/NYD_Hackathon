from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.upstash import UpstashVectorStore
from llama_index.core import StorageContext
from sentence_transformers import SentenceTransformer
import textwrap
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

Settings.llm = Groq(model="llama3-70b-8192", api_key="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")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
)

class UpstashVectorDatabase:
    def __init__(self, url, token):
        """Initialize Upstash vector database."""
        self.vector_store = UpstashVectorStore(
            url=url,
            token=token
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        self.embed_model = Settings.embed_model
        self.embed_model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_index_from_documents(self, documents):
        """Create a new index from documents."""
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
        return self.index

    def load_documents(self, csv_path):
        """Load documents from CSV file."""
        import pandas as pd

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Create documents from question, chapter, verse, and translation
        documents = []
        for _, row in df.iterrows():
            content = f"Chapter {row['chapter']}, Verse {row['verse']}: {row['translation']}"
            documents.append(Document(text=content))
        
        # Include questions for evaluation
        questions = df['question'].tolist()
        translations = df['translation'].tolist()
        
        return documents, questions, translations

    def query(self, query_text, wrap_width=100):
        """Query the vector database and return formatted response."""
        if not self.index:
            raise ValueError("No index available. Please create an index first.")
        
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        return textwrap.fill(str(response), wrap_width)

    def calculate_similarity(self, text1, text2):
        """Calculate cosine similarity between two sentences using embeddings."""
        embeddings = self.embed_model_similarity.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

def main():
    """Main function to demonstrate UpstashVectorDatabase usage."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db = UpstashVectorDatabase(
        url="https://champion-ibex-54536-us1-vector.upstash.io",
        token="ABcFMGNoYW1waW9uLWliZXgtNTQ1MzYtdXMxYWRtaW5ObVZtTVdSbVlUZ3RZek5tWXkwME1tUmxMVGcyTXpNdFpUZGxaV1ZoWkdaaU5UUmw="
    )
    
    try:
        logger.info("Loading documents from CSV...")
        documents, questions, translations = db.load_documents("Data/patanjali.csv")
        logger.info(f"Loaded {len(documents)} verses")

        logger.info("Creating vector index...")
        db.create_index_from_documents(documents)
        logger.info("Index created successfully")

        correct_count = 0
        total_queries = len(questions)
        similarity_threshold = 0.65

        logger.info("Evaluating queries...")
        for i, question in enumerate(questions):
            response = db.query(question)
            retrieved_answer = response.strip()
            expected_answer = translations[i].strip()

            # Calculate similarity score
            similarity = db.calculate_similarity(retrieved_answer, expected_answer)

            # Determine if the response is correct
            is_correct = similarity >= similarity_threshold
            if is_correct:
                correct_count += 1

            logger.info(f"\nQuestion: {question}")
            logger.info(f"Expected: {expected_answer}")
            logger.info(f"Retrieved: {retrieved_answer}")
            logger.info(f"Similarity: {similarity:.2f}")
            logger.info(f"Match: {'Yes' if is_correct else 'No'}")

        # Calculate accuracy
        accuracy = correct_count / total_queries * 100
        logger.info(f"Accuracy: {accuracy:.2f}%")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
