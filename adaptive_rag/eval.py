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
from ragas_evaluation import llm_eval
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILTY_API_KEY"] = "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"
load_dotenv()

import pandas as pd
import random

def evaluate_questions(df, workflow, llm, num_questions=20):
    gita_questions = df[df['book'] == "Bhagwad Gita"].sample(num_questions, random_state=42)
    patanjali_questions = df[df['book'] == "Patanjali Yoga Sutras"].sample(num_questions, random_state=42)

    results = []
    correct_verse_count = 0
    correct_chapter_count = 0

    for idx, row in pd.concat([gita_questions, patanjali_questions]).iterrows():
        ground_truth = row['augmented_response']
        verse_gt = str(row['verse'])
        chapter_gt = str(row['chapter'])
        question = str(row['question'])
        
        inputs = {"question": question}
        model_output = workflow.run_workflow(inputs)
        generated_text = model_output['generation']
        verses_info = model_output['extracted_info']
        documents = model_output['documents']
        retrieved_contents = []
        retrieved_contents.append(documents[0].page_content)
        retrieved_contents.append(documents[1].page_content)

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
        
        ragas_evaluation_result = llm_eval(rag_response, question, ground_truth, retrieved_contents)
        
        predicted_verse = verse_extractor(rag_response).strip()
        predicted_chapter = chapter_extractor(rag_response).strip()

        if predicted_verse == verse_gt:
            correct_verse_count += 1
        if predicted_chapter == chapter_gt:
            correct_chapter_count += 1

        rouge_l, bert_f1, cos_sim, bleu = calculate_evaluation_metrics(ground_truth, rag_response)

        print("RAGAS evaluation result: \n", ragas_evaluation_result)
        results.append({
            'book': row['book'],
            'ground_truth': ground_truth,
            'model_output': model_output,
            'verse_gt': verse_gt,
            'predicted_verse': predicted_verse,
            'chapter_gt': chapter_gt,
            'predicted_chapter': predicted_chapter,
            'rouge_l': rouge_l,
            'bert_f1': bert_f1,
            'cos_sim': cos_sim,
            'bleu': bleu,
            'context recall': ragas_evaluation_result['context_recall'],
            'faithfulness': ragas_evaluation_result['faithfulness'],
        })
        

    total_questions = len(results)
    verse_accuracy = correct_verse_count / total_questions
    chapter_accuracy = correct_chapter_count / total_questions

    print(f"Verse Extraction Accuracy: {verse_accuracy:.2f}")
    print(f"Chapter Extraction Accuracy: {chapter_accuracy:.2f}")

    return results, verse_accuracy, chapter_accuracy




def verse_extractor(answer):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"), 
        max_retries=10
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
    embd_model="BAAI/bge-small-en-v1.5"
    k = 3
    csv_path = r'..\Combined_Data\Merged_Bhagwad_Gita_and_Patanjali_Yoga_Sutras.csv'
    
    api_key = os.getenv("GROQ_API_KEY")
    
    
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        api_key=api_key
    )
    
    workflow = Workflow(model, embd_model, api_key, k, csv_path)
    
    df = pd.read_csv(csv_path)
    

    results, verse_accuracy, chapter_accuracy = evaluate_questions(df, workflow = workflow, llm = llm, num_questions=50)

    results_df = pd.DataFrame(results)
    
    print("Verse accuracy : ", verse_accuracy)
    print("Chapter accuracy : ", chapter_accuracy)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation complete. Results saved to evaluation_results.csv.")
    
    
if __name__ == "__main__":
    main()