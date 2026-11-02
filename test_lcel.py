
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 1. Create a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming tutor."),
    ("human", "Explain {topic} in three simple bullet points."),
])

# 2. Create a fake model that returns a predictable response
def fake_model(messages):
    print("\nPrompt sent to the model:")
    print(messages)

    return "• Point one\n• Point two\n• Point three"

model = RunnableLambda(fake_model)

# 3. Convert the output into a plain string
output_parser = StrOutputParser()

# 4. Compose the workflow using LCEL
chain = prompt | model | output_parser

# 5. Run the workflow
result = chain.invoke({"topic": "RAG"})
print("\nSecond result:")
print(result)

print("\nFinal result:")
print(result)
