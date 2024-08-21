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
