
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming tutor."),
    ("human", "Explain {topic} in two simple bullet points."),
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
output_parser = StrOutputParser()

answer_chain = prompt | model | output_parser

chain = RunnablePassthrough.assign(
    answer=answer_chain
)

topic = input("Enter a topic: ").strip()

if topic:
    result = chain.invoke({"topic": topic})
    print("\nOriginal topic:", result["topic"])
    print("\nAI answer:\n", result["answer"])
else:
    print("Please enter a topic.")
