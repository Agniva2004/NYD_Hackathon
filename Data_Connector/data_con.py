from langchain_groq import ChatGroq
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# List of API keys
api_keys = os.getenv("GROQ_API_KEYS").split(",")

# Function to get the next available API key
class APIKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0

    def get_key(self):
        if self.index >= len(self.keys):
            raise Exception("All API keys exhausted")
        key = self.keys[self.index]
        self.index += 1
        return key

key_manager = APIKeyManager(api_keys)

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=key_manager.get_key(),
)

def generate_augmented_data(row):
    user_input = (
        f"Chapter: {row['chapter']}, Verse: {row['verse']}, Speaker: {row['speaker']}\n"
        f"Translation: {row['translation']}\n"
        f"Question: {row['question']}"
    )
    
    system_prompt = (
        "You are an expert on the Bhagavad Gita, providing clear and accurate responses to questions. "
        "Based on the provided chapter, verse, speaker, and translation, generate a concise and precise answer to the question. "
        "Ensure the response is directly relevant to the question without unnecessary elaboration. "
        f"Here is the input for your reference:\n{user_input}"
    )
    
    try:
        # Generate response using the LLM
        response = llm.invoke(system_prompt)
        answer = response.content.strip()
        
        # Append the sloka after the generated response
        sloka = row['sanskrit']
        final_response = f"""{answer}\n\nThe sloka associated with this answer is:\n{sloka}"""
        
        return final_response
    except Exception as e:
        print(f"Error generating response for chapter {row['chapter']} verse {row['verse']}: {e}")
        try:
            # Switch to the next API key
            llm.api_key = key_manager.get_key()
            response = llm.invoke(system_prompt)
            answer = response.content.strip()
            sloka = row['sanskrit']
            final_response = f"""{answer}\n\nThe sloka associated with this answer is:\n{sloka}"""
            return final_response
        except Exception as e:
            print(f"Error with fallback key for chapter {row['chapter']} verse {row['verse']}: {e}")
            return None

def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    augmented_responses = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        augmented_response = generate_augmented_data(row)
        augmented_responses.append(augmented_response)
        print(augmented_response)

    # Assign augmented responses to the corresponding rows
    df["augmented_response"] = augmented_responses

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = r'C:\Users\Anushree\Desktop\NYD\NYD_Hackathon\Data\Bhagwad_Gita\Bhagwad_Gita_Verses_English_Questions.csv' 
    output_csv = r'C:\Users\Anushree\Desktop\NYD\NYD_Hackathon\Data\Bhagwad_Gita\Bhagwad_Gita_Verses_English_augmented_appended.csv'  

    process_csv(input_csv, output_csv)
