
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# 1. Initialize the Gemini chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# 2. Create a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful programming tutor. "
        "Explain concepts clearly and simply."
    ),
    (
        "human",
        "Explain {topic} in three simple bullet points."
    )
])

# 3. Convert the model's response into plain text
output_parser = StrOutputParser()

# 4. Connect the components into a chain
chain = prompt | model | output_parser

# 5. Run the chain
if __name__ == "__main__":
    topic = input("Enter a topic to learn: ")

    result = chain.invoke({"topic": topic})

    print("\nAnswer:\n")
    print(result)
