"""
Parse the config files.
"""

import json

with open(
    "/Users/al.farace/Projects/llm-interactive-story/config_openai.json",
    "r",
    encoding="utf-8",
) as f:
    openai_config = json.load(f)

with open(
    "/Users/al.farace/Projects/llm-interactive-story/config_local.json",
    "r",
    encoding="utf-8",
) as f:
    local_config = json.load(f)
