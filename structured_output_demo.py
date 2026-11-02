from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# 1. Define the required output structure
class SupportTicket(BaseModel):
    category: Literal[
        "payment", "delivery", "technical", "other"
    ] = Field(
        description="The category of the support request"
    )

    priority: Literal[
        "low", "medium", "high"
    ] = Field(
        description="The priority of the support request"
    )


# 2. Configure the model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# 3. Ask the model for structured output
structured_model = model.with_structured_output(SupportTicket)

# 4. Define the prompt
prompt = ChatPromptTemplate.from_template("""
Classify this customer support request.

Choose a category from: payment, delivery, technical, other.
Choose a priority from: low, medium, high.

Support request:
{ticket}
""")

# 5. Connect the components
chain = prompt | structured_model

# 6. Test the chain
# 6. Test the chain with error handling
ticket_text = (
    "My payment was deducted, but my order "
    "wasn't confirmed."
)

try:
    # Configure automatic retries
    retry_chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# Invoke the chain with retries enabled
    result = retry_chain.invoke({
    "ticket": ticket_text
})
    print(result)
    print("\nCategory:", result.category)
    print("Priority:", result.priority)

except Exception as error:
    print("Something went wrong while processing the ticket.")
    print("Error type:", type(error).__name__)
    print("Error details:", error)