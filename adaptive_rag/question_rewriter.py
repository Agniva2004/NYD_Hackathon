from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
class QuestionRewriter:
    def __init__(self, model, api_key):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt =  """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        self.re_write_question_ = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        self.question_rewriter = self.re_write_question_ | self.llm | StrOutputParser()
    def re_write_question(self, question):
        better_question = self.question_rewriter.invoke({"question": question})
        return better_question
    
    