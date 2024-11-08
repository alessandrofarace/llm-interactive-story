"""
Parse the config files.
"""

import json
import os
from dataclasses import dataclass, field


@dataclass
class StoryConfig:
    n_continuations: int

    def __post_init__(self):
        if self.n_continuations < 0 or self.n_continuations > 3:
            raise ValueError(
                "Currently only 1, 2 or 3 continuations for each chapter are supported."
            )


@dataclass
class LLMConfig:
    openai_client_params: dict = field(default_factory=dict)
    chat_completion_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if "api_key" in self.openai_client_params:
            self.openai_client_params["api_key"] = self._get_key(
                self.openai_client_params["api_key"]
            )

    def _get_key(self, key: str) -> str:
        if key == "GROQ":
            return os.environ["GROQ_API_KEY"]
        else:
            return key


@dataclass
class IllustratorConfig:
    prompter_config: LLMConfig
    painter_config: dict = field(default_factory=dict)


def parse_llm_config(params_dict: dict, default_config: LLMConfig) -> LLMConfig:
    llm_config = LLMConfig(**params_dict)
    if llm_config.openai_client_params == "default":
        llm_config.openai_client_params = default_config.openai_client_params
    if llm_config.chat_completion_params == "default":
        llm_config.chat_completion_params = default_config.chat_completion_params
    return llm_config


with open(
    "/Users/al.farace/Projects/llm-interactive-story/config/config.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)


story_config = StoryConfig(**config["story_config"])

default_llm_config = LLMConfig(
    openai_client_params=config["default_openai_client_params"],
    chat_completion_params=config["default_chat_completion_params"],
)

storyteller_config = parse_llm_config(
    params_dict=config["storyteller_config"],
    default_config=default_llm_config,
)

translator_config = parse_llm_config(
    params_dict=config["translator_config"],
    default_config=default_llm_config,
)

illustrator_config = IllustratorConfig(
    prompter_config=parse_llm_config(
        params_dict=config["illustrator_config"]["prompter_config"],
        default_config=default_llm_config,
    ),
    painter_config=config["illustrator_config"]["painter_config"],
)
