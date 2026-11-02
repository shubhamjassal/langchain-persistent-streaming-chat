
from pydantic import ValidationError
from langchain_core.runnables import RunnableBranch, RunnableLambda

from routing_models import RouteInput


def invalid_route_response(data):
    return {
        "answer": (
            f"Invalid route type: {data.get('type')!r}. "
            "Expected 'document' or 'general'."
        ),
        "source": "router",
        "status": "invalid_route",
        "sources": [],
    }


def invalid_input_response(error):
    return {
        "answer": f"Invalid input: {error}",
        "source": "router",
        "status": "invalid_input",
        "sources": [],
    }


def validate_input(data):
    try:
        validated = RouteInput.model_validate(data)
        return validated.model_dump()
    except ValidationError as error:
        return invalid_input_response(error)


def build_router(document_chain, general_chain):
    return (
        RunnableLambda(validate_input)
        | RunnableBranch(
            (
                lambda data: data.get("status") == "invalid_input",
                RunnableLambda(lambda data: data),
            ),
            (
                lambda data: data["type"] == "document",
                document_chain,
            ),
            (
                lambda data: data["type"] == "general",
                general_chain,
            ),
            RunnableLambda(invalid_route_response),
        )
    )
