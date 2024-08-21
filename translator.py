import logging

import json_repair
import openai

from config import config

logger = logging.getLogger(__name__)

translator_config = config["translator_config"]


class LLMTranslator:

    client = openai.OpenAI(**translator_config["openai_client_config"])
    system_prompt = """
* You are an expert translator from English to {language}.
* You are particularly good at translating children stories.
* You MUST avoid mature themes like violence, guns, drugs, alcohol and sex.

You MUST output only the requested translation, without any additional comments.
You only answer in {language}.
"""

    def __init__(self, language: str = "Italian") -> None:
        self.language = language
        self.system_prompt = self.system_prompt.format(language=language)

    def translate(self, text: str) -> str:
        translation_prompt = """
Translate the following text:

{text}
"""
        if self.language == "English":
            return text
        else:
            message_history = [
                {"role": "system", "content": self.system_prompt},
                {
                    "name": "user",
                    "role": "user",
                    "content": translation_prompt.format(text=text),
                },
            ]
            completion = self.client.chat.completions.create(
                messages=message_history,
                **translator_config["chat_completion_params"],
            )
            model_response = completion.choices[0].message.content
            logger.info("translation: %s", model_response)
            return model_response
