from langchain_core.runnables import RunnableLambda


# Simulate an operation that fails twice, then succeeds
attempts = {"count": 0}


def unreliable_operation(value):
    attempts["count"] += 1
    print(f"Attempt {attempts['count']}")

    if attempts["count"] < 3:
        raise ConnectionError("Temporary connection failure")

    return f"Success: {value}"


# Wrap the function as a LangChain runnable
chain = RunnableLambda(unreliable_operation)

# Enable retries
retry_chain = chain.with_retry(
    retry_if_exception_type=(ConnectionError,),
    stop_after_attempt=3,
    wait_exponential_jitter=False,
)

try:
    result = retry_chain.invoke("Hello LangChain")
    print(result)

except ConnectionError as error:
    print("All retry attempts failed:", error)