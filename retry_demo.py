import time


def unreliable_operation():
    """Simulate an operation that fails twice, then succeeds."""
    unreliable_operation.attempts += 1

    print(f"Attempt {unreliable_operation.attempts}")

    if unreliable_operation.attempts < 3:
        raise ConnectionError("Temporary connection failure")

    return "Operation successful!"


unreliable_operation.attempts = 0

max_attempts = 4

for attempt in range(1, max_attempts + 1):
    try:
        result = unreliable_operation()
        print(result)
        break

    except ConnectionError as error:
        print(error)

        if attempt == max_attempts:
            print("All attempts failed.")
            break

        delay = 2 ** (attempt - 1)
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)