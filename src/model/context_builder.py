from pydantic import BaseModel

from .models import Function, Prompt


class ContextBuilder(BaseModel):
    functions: list[Function]

    def build(self) -> str:
        context: str = ""
        for function in self.functions:
            context += (
                f"Function: {function.name}\n"
                f"Description: {function.description}\n\n"
            )
        return context

    def build_with_prompt(self, prompt: Prompt) -> str:
        context: str = self.build()
        context += f"Prompt: {prompt.prompt}\n"
        return context
