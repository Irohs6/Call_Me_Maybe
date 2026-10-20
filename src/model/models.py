from pydantic import BaseModel
from pydantic import ConfigDict


class Type(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Type]
    returns: Type


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
