
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming tutor."),
    ("human", "Explain {topic} in three simple bullet points."),
])

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,

)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

if __name__ == "__main__":
    topic = input("Enter a topic to learn: ").strip()

    if not topic:
        print("Please enter a topic.")
    else:
        try:
            result = chain.invoke({"topic": topic})
            print("\nAnswer:\n")
            print(result)
        except Exception as error:
            print(f"Request failed: {type(error).__name__}")
            print("Check your API quota, model availability, and API configuration.")
