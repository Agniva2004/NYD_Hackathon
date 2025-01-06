from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )    
    
class AnswerGrader:
    def __init__(self, model, api_key):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.structured_llm_grader = self.llm.with_structured_output(GradeAnswer)
        self.system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        self.answer_grader = self.answer_prompt | self.structured_llm_grader
    def grade_answer(self, question, generation):
        score = self.answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        return grade