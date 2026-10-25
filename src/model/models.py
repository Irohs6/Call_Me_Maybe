from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Type(BaseModel):
    """Describe an allowed argument or return type."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "boolean"]


class Function(BaseModel):
    """Describe a callable function from the input catalogue."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="The name of the function")
    description: str = Field(
        min_length=1, description="The description of the function"
    )
    parameters: dict[str, Type]
    returns: Type


class Prompt(BaseModel):
    """Store one user request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="The prompt text")


class FunctionCall(BaseModel):
    """Store the generated function name and arguments."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
