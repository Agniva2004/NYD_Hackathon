import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ["TAVILY_API_KEY"] = "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"
from workflow import Workflow
from adaptive_rag_class import LoadDocuments
from rouge import Rouge
from bert_score import score as bert_score  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For BLEU score (`pip install nltk`)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ReactAgent.react_agent import React_Agent
def calculate_evaluation_metrics(ground_truth, model_output):

    rouge = Rouge()
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    smoothing_function = SmoothingFunction().method4

    # ROUGE-L Score
    rouge_scores = rouge.get_scores(model_output, ground_truth)
    rouge_l_score = rouge_scores[0]['rouge-l']['f']

    # BERTScore
    P, R, F1 = bert_score([model_output], [ground_truth], lang="en")
    bert_f1 = F1.mean().item()

    # Cosine Similarity
    embeddings = sentence_transformer.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # BLEU Score
    reference = ground_truth.split()
    candidate = model_output.split()
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)

    # Print scores
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
    api_key = "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"
    k = 3
    csv_path = r'C:\Users\Srinjoy\OneDrive\Desktop\NYD\NYD_Hackathon\Data\Patanjali_Yoga_Sutras\Patanjali_Yoga_Sutras_Verses_English_Questions_augmented_appended.csv'
    
    
    
    workflow = Workflow(model, embd_model, api_key, k, csv_path)
    # evaluation_csv_path = 'Data\sampled_patanjali_yoga_sutras.csv'
    # load_documents = LoadDocuments(evaluation_csv_path)
    # documents, questions, translations = load_documents.load_documents()
    # inputs = {
    #     "question": "What is the purpose of yoga?"
    # }

    question = "What is wrong knowledge?"
    inputs = {"question": question}
    model_output = workflow.run_workflow(inputs)
    ground_truth = "False knowledge arises from misperception, which is not based on reality."
        
    average_rouge_l, average_bert_score, average_cosine_similarity, average_bleu_score = calculate_evaluation_metrics(ground_truth, model_output) 
    
    
    
    
if __name__ == "__main__":
    main()                
        