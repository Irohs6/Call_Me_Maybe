from abc import ABC, abstractmethod

from llm_sdk import Small_LLM_Model

from .model.models import Function, Prompt


class ArgumentGenerator(ABC):

    def __init__(
        self,
        model: Small_LLM_Model,
    ):
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameter: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        pass
