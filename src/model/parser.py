from pydantic import BaseModel, Field, model_validator, ValidationError
import json


class JsonParser(BaseModel):
    data: str

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def parse(cls, data: str) ->JsonParser :
        try:


class TypeDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition
