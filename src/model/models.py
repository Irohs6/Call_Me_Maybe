from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any


class Type(BaseModel):
    """Describe the JSON type of a function parameter or return."""

    model_config = ConfigDict(extra="forbid")

    type: str


class Function(BaseModel):
    """Describe an available function and its parameter schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="The name of the function")
    description: str = Field(min_length=1, description="The description"
                             "of the function")
    parameters: dict[str, Type]
    returns: Type

    @field_validator("parameters")
    @classmethod
    def check_parameters(
        cls, parameters: dict[str, Type],
    ) -> dict[str, Type]:
        """Reject empty names and unsupported argument types."""
        allowed = {"string", "number", "integer", "boolean"}
        for name, definition in parameters.items():
            if not name.strip():
                raise ValueError("Parameter name cannot be empty")
            if definition.type not in allowed:
                raise ValueError(
                    f"Unsupported type for '{name}': {definition.type}"
                )
        return parameters


class Prompt(BaseModel):
    """Represent a natural-language function-calling request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="The prompt text")


class FunctionCall(BaseModel):
    """Represent one function call in the output JSON array."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
