import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dotenv import load_dotenv
from workflow import Workflow
import time
from adaptive_rag_class import LoadDocuments
# from rouge import Rouge
# from bert_score import score as bert_score  
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For BLEU score (`pip install nltk`)
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from prompt_optimizer.poptim import EntropyOptim
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILTY_API_KEY"] = "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"
load_dotenv()
import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


set_llm_cache(GPTCache(init_gptcache))
# def calculate_evaluation_metrics(ground_truth, model_output):

#     rouge = Rouge()
#     sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     smoothing_function = SmoothingFunction().method4

#     rouge_scores = rouge.get_scores(model_output, ground_truth)
#     rouge_l_score = rouge_scores[0]['rouge-l']['f']

#     P, R, F1 = bert_score([model_output], [ground_truth], lang="en")
#     bert_f1 = F1.mean().item()

#     embeddings = sentence_transformer.encode([model_output, ground_truth])
#     cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

#     reference = ground_truth.split()
#     candidate = model_output.split()
#     bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)

#     print(f"Ground Truth: {ground_truth}")
#     print(f"Model Output: {model_output}")
#     print(f"ROUGE-L Score: {rouge_l_score:.4f}")
#     print(f"BERTScore (F1): {bert_f1:.4f}")
#     print(f"Cosine Similarity: {cos_sim:.4f}")
#     print(f"BLEU Score: {bleu:.4f}\n")

#     return rouge_l_score, bert_f1, cos_sim, bleu
  
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
    
    start = time.time()

    question = "What are the profound ways in which spiritual practices, such as meditation, prayer, or mindfulness, can help individuals cultivate inner peace, resilience, and a deeper connection to their purpose in life?"
    
    word_count = len(question.split())
    p_optimizer = EntropyOptim(verbose=True, p=0.1)
    if word_count > 30:
        question = p_optimizer(question)['content']
        print("Optimized question: ", question)
    inputs = {"question": question}
    model_output = workflow.run_workflow(inputs)
    
    generated_text = model_output['generation']
    verses_info = model_output['extracted_info']
    
    messages = [
    (
        "system",
        "You are a helpful assistant designed to combine a generated response with relevant verse information. Your task is to create a complete, coherent, and concrete answer. \
        The input consists of two parts: \
        1. **Generated Response**: This is the model-generated answer to the user's query. \
        2. **Extracted Text**: This contains specific verse information relevant to the query. \
        Your job is to integrate the extracted text into the generated response seamlessly, ensuring the final output is meaningful and provides comprehensive information. \
        Use formal language and ensure the response is well-structured and clear."
    ),
    ("human", "Generated Response: " + generated_text + "\n" + "Extracted Text: " + verses_info),
]

    rag_response = llm.invoke(messages)
    rag_response = rag_response.content
    # ground_truth = "False knowledge arises from misperception, which is not based on reality."
    
    end = time.time()
    print("Time taken : ", end - start)
        
    # average_rouge_l, average_bert_score, average_cosine_similarity, average_bleu_score = calculate_evaluation_metrics(ground_truth, rag_response) 
    print("Pipeline response: \n" , rag_response)
    
    
    
if __name__ == "__main__":
    main()                
        