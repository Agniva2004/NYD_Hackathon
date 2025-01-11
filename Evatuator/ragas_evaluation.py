import pandas as pd
import random
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
import os

def evaluate_dataset(data_path, sample_size=10, model="mixtral-8x7b-32768"):
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set in the .env file.")

    llm = ChatGroq(
        model=model,
        api_key=groq_api_key,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    df = pd.read_csv(data_path)

    if not {"question", "augmented_response", "translation"}.issubset(df.columns):
        raise ValueError("The dataset must contain 'question', 'augmented_response', and 'translation' columns.")

    sampled_data = df.sample(n=sample_size, random_state=42) 

    dataset = []

    for _, row in sampled_data.iterrows():
        query = row["question"]
        response = row["augmented_response"]
        reference = row["translation"]

        dataset.append(
            {
                "user_input": query,
                "response": response,           
                "reference": reference,
                "retrieved_contexts": [reference]        
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )

    print("Evaluation Results:")
    print(result)

data_path = r"..\Combined_Data\Merged_Bhagwad_Gita_and_Patanjali_Yoga_Sutras.csv"
evaluate_dataset(data_path)
