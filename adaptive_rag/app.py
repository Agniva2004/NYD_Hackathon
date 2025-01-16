import os
import time
import joblib
import streamlit as st
from workflow import Workflow
from langchain_groq import ChatGroq
import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

MODEL_ROLE = "ai"
AI_AVATAR_ICON = "🧘‍♂️"
DATA_FOLDER = "data/"
os.makedirs(DATA_FOLDER, exist_ok=True)

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )

set_llm_cache(GPTCache(init_gptcache))

def initialize_workflow():
    model = "llama3-70b-8192"
    embd_model = "sentence-transformers/all-MiniLM-L6-v2"
    api_key = os.getenv("GROQ_API_KEY")
    k = 3
    csv_path = r'..\Combined_Data\Merged_Bhagwad_Gita_and_Patanjali_Yoga_Sutras.csv'
    return Workflow(model, embd_model, api_key, k, csv_path)


st.set_page_config(page_title="Yogi Bot", page_icon="🧘‍♂️", layout="wide")
st.title("🧘‍♂️ MARSA: Patanjali Yoga Sutras & Bhagwad Gita Chatbot")
st.write("Ask questions about the Patanjali Yoga Sutras and Bhagwad Gita to receive insightful responses.")



workflow = initialize_workflow()


st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    inputs = {"question": prompt}
    model_output = workflow.run_workflow(inputs)
    generated_text = model_output["generation"]
    verses_info = model_output["extracted_info"]

    messages = [
        ("system", "You are a helpful assistant designed to combine a generated response with relevant verse information. "
                   "Your task is to create a complete, coherent, and concrete answer. The input consists of two parts: "
                   "1. **Generated Response**: This is the model-generated answer to the user's query. "
                   "2. **Extracted Text**: This contains specific verse information relevant to the query. "
                   "Your job is to integrate the extracted text into the generated response seamlessly, ensuring the final output "
                   "is meaningful and provides comprehensive information."),
        ("human", f"Generated Response: {generated_text}\nExtracted Text: {verses_info}")
    ]

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, max_tokens=None)
    rag_response = llm.invoke(messages).content

    with st.chat_message(MODEL_ROLE, avatar=AI_AVATAR_ICON):
        st.markdown(rag_response)

    st.session_state.messages.append({"role": MODEL_ROLE, "content": rag_response, "avatar": AI_AVATAR_ICON})


