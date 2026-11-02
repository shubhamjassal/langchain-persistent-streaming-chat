
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming tutor."),
    (
        "human",
        "Explain {topic} for a {level} learner "
        "in exactly {points} bullet points."
    ),
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

user_input = {
    "topic": "embeddings",
    "level": "beginner",
    "points": 3,
}

result = chain.invoke(user_input)

print("\nAnswer:\n")
print(result)
