"""
Parse the config files.
"""

import json

with open(
    "/Users/al.farace/Projects/llm-interactive-story/config_openai.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)

openai_config = config["openai_config"]
completion_params = config["completion_params"]
