from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
class Planner_agent:
    def __init__(self, model, api_key, question, current_answer):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt =  "You are an AI planner expert in restructuring the answer in a way that can completely resolve the question. The original question failed to be resolved by the current answer. "
        "Follow these steps to regenerate the question:\n"
        "1. Analyze the original question for ambiguities or missing information.\n"
        "2. Evaluate the current answer to identify why it failed.\n"
        "3. Break down the problem into key components or sub-questions.\n"
        "4. Regenrate the answer using a clear, chain-of-thought approach to ensure it resolves ambiguities.\n"
        "5. Keep the response concise and to the point.\n"
        "6. Validate the final answer.\n\n"
        f"Original Question: {question}\n"
        f"Current Answer: {current_answer}\n\n"
        "Regenerated answer:"
        self.planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Here is the original question: \n\n {question} \n\n Here is the current answer: \n\n {current_answer} \n\n Regenerate the answer.",
                ),
            ]
        )
        self.planner = self.planner_prompt | self.llm | StrOutputParser()
    def planner(self, question):
        better_answer = self.planner.invoke({"question": question})
        return better_answer  
        