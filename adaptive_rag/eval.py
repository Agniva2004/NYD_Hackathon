import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dotenv import load_dotenv
from workflow import Workflow
from adaptive_rag_class import LoadDocuments
from rouge import Rouge
from bert_score import score as bert_score  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For BLEU score (`pip install nltk`)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILTY_API_KEY"] = "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"
load_dotenv()


def verse_extractor(answer):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    messages = [
        ("system", """You are an expert who extracts the verse from the given text.
                        This is the sample example : 
                        Question : According to Chapter 1, Verse 2 of the Bhagavad Gita, Duryodhana first talked to his teacher, Drona.
                        Answer : 2
                        Question : According to Bhagavad Gita, Chapter 1, Verse 12, Bhishma blew his conch to cheer up Duryodhana.
                        Answer : 12
                        
                    You will only give the verse number, no more text is required.
         """),
        ("user", answer)
    ]
    response = llm.invoke(messages)
    return response.content

def chapter_extractor(answer):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    messages = [
        ("system", """You are an expert who extracts the chapter number from the given text.
                        This is the sample example : 
                        Question : According to Chapter 1, Verse 2 of the Bhagavad Gita, Duryodhana first talked to his teacher, Drona.
                        Answer : 1
                        Question : According to Bhagavad Gita, Chapter 1, Verse 12, Bhishma blew his conch to cheer up Duryodhana.
                        Answer : 1
                        
                    You will only give the chapter number, no more text is required.
         """),
        ("user", answer)
    ]
    response = llm.invoke(messages)
    return response.content


def calculate_evaluation_metrics(ground_truth, model_output):

    rouge = Rouge()
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    smoothing_function = SmoothingFunction().method4

    rouge_scores = rouge.get_scores(model_output, ground_truth)
    rouge_l_score = rouge_scores[0]['rouge-l']['f']

    P, R, F1 = bert_score([model_output], [ground_truth], lang="en")
    bert_f1 = F1.mean().item()

    embeddings = sentence_transformer.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    reference = ground_truth.split()
    candidate = model_output.split()
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)

    print(f"Ground Truth: {ground_truth}")
    print(f"Model Output: {model_output}")
    print(f"ROUGE-L Score: {rouge_l_score:.4f}")
    print(f"BERTScore (F1): {bert_f1:.4f}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print(f"BLEU Score: {bleu:.4f}\n")

    return rouge_l_score, bert_f1, cos_sim, bleu


def main():
    model = "llama3-70b-8192"
    embd_model="sentence-transformers/all-MiniLM-L6-v2"
    k = 3
    csv_path = r'..\Combined_Data\Merged_Bhagwad_Gita_and_Patanjali_Yoga_Sutras.csv'
    
    api_key = os.getenv("GROQ_API_KEY")
    
    
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        api_key=api_key
    )
    
    workflow = Workflow(model, embd_model, api_key, k, csv_path)