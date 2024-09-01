from typing import Any

from pydantic import BaseModel, Field, validator


class ResponseValidator(BaseModel):
    context: str = Field(description="The context from knoweldge base")
    query: str = Field(description="The user query for the LLM")
    result: str = Field(
        description="Response to the user query based on knowledge base"
    )
    source_documents: list[Any] = Field(
        description="List of sources from knowledge base"
    )
    confidence: dict[str, float] = Field(
        description="Confidence metrics using token log probabilities"
    )

    @validator("result")
    def validate_string(cls, value):
        if not isinstance(value, str):
            raise ValueError(
                f"Response from LLM is not valid! Expected string but got {type(value)}"
            )
        return value
