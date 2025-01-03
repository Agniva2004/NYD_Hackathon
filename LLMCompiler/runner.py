from langGraphChain import GraphChain
from tools import create_search_tool, create_yoga_sutras_tool
from plan_scheduler import plan_and_schedule
from langchain import hub
import groq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from joiner import FinalResponse, Replan, JoinOutputs
from joiner import _parse_joiner_output, select_recent_messages
from langchain_groq import ChatGroq
from langGraphChain import GraphChain
import os

llm = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

search_tool = create_search_tool()
retriever_tool = create_yoga_sutras_tool()

tools = [search_tool]

tool_messages = plan_and_schedule.invoke(
    {"messages": [HumanMessage(content="Define Imagination? What is imagination? What is fantasy?")]}
)["messages"]

joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  

runnable = joiner_prompt | llm.with_structured_output(JoinOutputs)


joiner = select_recent_messages | runnable | _parse_joiner_output

retriver_tool_response = retriever_tool.invoke("Define Imagination? What is imagination? What is fantasy?")

input_messages = [HumanMessage(content="Define Imagination? What is imagination? What is fantasy?")] + tool_messages + [retriver_tool_response]

try:
    answer = joiner.invoke({"messages": input_messages})
except groq.BadRequestError as err:
    error_details = err.response.json()['error']
    if 'failed_generation' in error_details:
        failed_gen = error_details['failed_generation']
        if 'Thought:' in failed_gen:
            thought = failed_gen.split('Thought:')[1].split('Action:')[0].strip()
            print(f"Final Answer: {thought}")
    else:
        raise err
else:
    print(f"Final Answer: {answer['messages'][-1].content}")

# chain = GraphChain().return_chain()

# for step in chain.stream(
#     {"messages": [HumanMessage(content="What is the meaning of life?")]}
# ):
#     print(step)
#     print("---")
#     break
    
# print(step["plan_and_schedule"]["messages"][-1].content)

