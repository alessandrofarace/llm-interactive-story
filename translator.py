import logging

from config import config
from llm_utils import LLMAgent

logger = logging.getLogger(__name__)

translator_config = config["translator_config"]


class LLMTranslator(LLMAgent):

    config = translator_config
    system_prompt = """
You are an expert translator from English to {language}.
You are particularly good at translating children stories.
You hate doing grammar mistakes and always double check your translation.
You MUST avoid mature themes like violence, guns, drugs, alcohol and sex.

You MUST output only the requested translation, without any additional comments.
"""

    def __init__(self, language: str = "Italian") -> None:
        super().__init__()
        self.language = language
        self.system_prompt = self.system_prompt.format(language=language)

    def translate(self, text: str) -> str:
        user_prompt_template = """
# Task
Carefully translate the following text:
{text}

You MUST output only the requested translation, without any additional comments.
"""
        if self.language == "English":
            return text
        else:
            user_prompt = user_prompt_template.format(text=text)
            message_history = [
                {"role": "system", "content": self.system_prompt},
                {
                    "name": "user",
                    "role": "user",
                    "content": user_prompt,
                },
            ]
            model_response = self.get_chat_completion_content(messages=message_history)
            logger.info("translation: %s", model_response)
            return model_response
