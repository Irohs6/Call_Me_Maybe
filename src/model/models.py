from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import Any


class Type(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="The name of the function")
    description: str = Field(min_length=1, description="The description"
                             "of the function")
    parameters: dict[str, Type]
    returns: Type


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="The prompt text")


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
