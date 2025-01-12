from experiment_agent import RetrievalAgent
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_word_overlap(original, generated):
    """Compute the word overlap percentage."""
    original_set = set(original.split())
    generated_set = set(generated.split())
    overlap = original_set.intersection(generated_set)
    return len(overlap) / len(original_set) if original_set else 0.0



def evaluate_metrics(original_answer, final_answer, vectorizer):
    if not original_answer or not final_answer:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    bleu = sentence_bleu([original_answer.split()], final_answer.split())

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(original_answer, final_answer)['rougeL'].fmeasure

    bert = bert_score([final_answer], [original_answer], lang="en")[2].item()

    vectors = vectorizer.transform([original_answer, final_answer])
    cosine_sim = cosine_similarity(vectors)[0, 1]

    word_overlap = compute_word_overlap(original_answer, final_answer)

    return bleu, rouge, bert, cosine_sim, word_overlap

agent_retrieval = RetrievalAgent()
agent_retrieval.setup_yoga_sutras_tool("../Data")
worker = agent_retrieval.setup_agent()
agent = worker.as_agent()

vectorizer = TfidfVectorizer()

test_files = [
    "..\Test_Data\sampled_Bhagwad_Gita_Verses.csv",
    "..\Test_Data\sampled_Patanjali_Yoga_Sutras.csv"
]

for file in test_files:
    df = pd.read_csv(file)
    df = df.dropna(subset=["question", "translation"])  
    vectorizer.fit(df["translation"]) 

    total_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
    num_questions = len(df)

    for index, row in df.iterrows():
        question = row["question"]
        original_answer = row["translation"]

        print(f"Question: {question}")
        final_answer = str(agent.chat(question))  

        bleu, rouge, bert, cosine_sim, word_overlap = evaluate_metrics(
            original_answer, final_answer, vectorizer
        )

        print(f"BLEU: {bleu}, ROUGE: {rouge}, BERTScore: {bert}, Cosine: {cosine_sim}, Word Overlap: {word_overlap}")

        total_metrics = [x + y for x, y in zip(total_metrics, [bleu, rouge, bert, cosine_sim, word_overlap])]

    avg_metrics = [x / num_questions for x in total_metrics]
    print(f"Average Metrics for {file}: BLEU: {avg_metrics[0]}, ROUGE: {avg_metrics[1]}, BERTScore: {avg_metrics[2]}, Cosine: {avg_metrics[3]}, Word Overlap: {avg_metrics[4]}")
