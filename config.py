"""
Parse the config files.
"""

import json

with open(
    "/Users/al.farace/Projects/llm-interactive-story/config.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)

story_config = config["story_config"]
storyteller_config = config["storyteller_config"]
translator_config = config["translator_config"]
illustrator_config = config["illustrator_config"]
