"""Base interface for typed argument generators."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from .constrained_decoding import ConstrainedDecoder
from .model.models import Function, Prompt


class ArgumentGenerator(BaseModel, ABC):
    """Define the common interface used by typed argument generators."""

    model_config = ConfigDict(extra="forbid")

    decoder: ConstrainedDecoder

    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate the requested subset of function arguments."""
        raise NotImplementedError
