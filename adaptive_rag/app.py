import os
import time
import joblib
import streamlit as st
from workflow import Workflow
from langchain_groq import ChatGroq

os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

MODEL_ROLE = "ai"
AI_AVATAR_ICON = "🧘‍♂️"
DATA_FOLDER = "data/"

os.makedirs(DATA_FOLDER, exist_ok=True)

def initialize_workflow():
    model = "llama3-70b-8192"
    embd_model = "BAAI/bge-small-en-v1.5"
    api_key = os.getenv("GROQ_API_KEY")
    k = 3
    csv_path = r'..\Combined_Data\Merged_Bhagwad_Gita_and_Patanjali_Yoga_Sutras.csv'
    return Workflow(model, embd_model, api_key, k, csv_path)

try:
    past_chats = joblib.load(os.path.join(DATA_FOLDER, "past_chats_list"))
except:
    past_chats = {}
    
st.set_page_config(page_title="Yogi Bot", page_icon="🧘‍♂️", layout="wide")

with st.sidebar:
    st.write("# Past Chats")
    new_chat_id = f"{time.time()}"
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = st.selectbox(
            label="Pick a past chat",
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, "New Chat"),
            placeholder="_",
        )
    else:
        st.session_state.chat_id = st.selectbox(
            label="Pick a past chat",
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, "New Chat" if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder="_",
        )
    st.session_state.chat_title = f"ChatSession-{st.session_state.chat_id}"


st.title("🧘‍♂️ Patanjali Yoga Sutras and Bhagwad Gita Chatbot")
st.write("Ask questions about the Patanjali Yoga Sutras, Bhagwad Gita and get insightful responses!")


workflow = initialize_workflow()

try:
    st.session_state.messages = joblib.load(os.path.join(DATA_FOLDER, f"{st.session_state.chat_id}-messages"))
except:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

if prompt := st.chat_input("Your question here..."):
    if st.session_state.chat_id not in past_chats:
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, os.path.join(DATA_FOLDER, "past_chats_list"))

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        dict(role="user", content=prompt)
    )

    inputs = {"question": prompt}
    model_output = workflow.run_workflow(inputs)
    generated_text = model_output['generation']
    verses_info = model_output['extracted_info']

    messages = [
        ("system", "You are a helpful assistant designed to combine a generated response with relevant verse information. Your task is to create a complete, coherent, and concrete answer. The input consists of two parts: 1. **Generated Response**: This is the model-generated answer to the user's query. 2. **Extracted Text**: This contains specific verse information relevant to the query. Your job is to integrate the extracted text into the generated response seamlessly, ensuring the final output is meaningful and provides comprehensive information. Use formal language and ensure the response is well-structured and clear."),
        ("human", f"Generated Response: {generated_text}\nExtracted Text: {verses_info}")
    ]

    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    rag_response = llm.invoke(messages).content

    with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
        st.markdown(rag_response)

    st.session_state.messages.append(
        dict(role=MODEL_ROLE, content=rag_response, avatar=AI_AVATAR_ICON)
    )

    joblib.dump(
        st.session_state.messages,
        os.path.join(DATA_FOLDER, f"{st.session_state.chat_id}-messages")
    )

# Display chat history section
# if st.session_state.messages:
#     st.write("### Chat History:")
#     for i, interaction in enumerate(st.session_state.messages):
#         if interaction['role'] == 'user':
#             st.write(f"**Q{i + 1}:** {interaction['content']}")
#         else:
#             st.write(f"**A{i + 1}:** {interaction['content']}")
#         st.write("---")
