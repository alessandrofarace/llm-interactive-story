import json
import logging
import random

import json_repair

from config import storyteller_config
from llm_utils import LLMAgent

logger = logging.getLogger(__name__)


# * You will be asked to write chapters of a story.
# * You MUST output one chapter at a time.
# * Keep each chapter simple and short.
# * After three of four chapters, you MUST start to close loose ends and converge to a conclusion.
# * When you are ready to finish the story, add the following words "THE END".
# * Otherwise do not add anything.

# You MUST output only the story, no comments like "here the story begins" or "here the chapter ends".


class LLMStoryteller(LLMAgent):

    config = storyteller_config
    system_prompt = """You are a storyteller for small children.
You tell short instructive stories.
You can set the stories in both realistic and fantastic worlds.
You should mix a few emojis inbetween words to add visual elements.
You MUST avoid mature themes like violence, guns, drugs, alcohol and sex.

You stick to the instructions given and you NEVER add comments and additional explanations."""

    def generate_protagonist(self) -> str:
        user_prompt = """Give me a list of 30 possible protagonists for a story for children.
Be creative. For example, the protagonist can be a boy or a girl, a human, a speaking animal, or a fantastical creature.
Output the list in JSON format. Do NOT output anything beside the json.

Example:
[
    "Harry the pirate",
    "Nemo the small fish",
    "Greta the mermaid",
    "Leo the curious boy",
    "Mary the fast girl",
    "Gaia the bee",
    "Sophie the veterinarian",
    "..."
]"""

        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("choosing protagonist")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("possible protagonists: %s", model_response)
        protagonists = json_repair.loads(model_response)
        choice = random.choice(protagonists)
        logger.info("choice: %s", choice)
        return choice

    def identify_protagonist(self, abstract: str) -> str:
        user_prompt_template = """The premise of the story is: {abstract}.
        
If a protagonist is mentioned, output the protagonist.
Otherwise identify a suitable protagonist that fits the setting given in the premise. In the second case be creative. For example, the protagonist can be a boy or a girl, a human, a speaking animal, or a fantastical creature.

Output only the protagonist without adding any comment. Format your output as in the examples below.

Examples:
"Harry the pirate",
"Nemo the small fish",
"Greta the mermaid",
"Leo the curious boy",
"Mary the fast girl",
"Gaia the bee",
"Sophie the veterinarian",
"""

        user_prompt = user_prompt_template.format(
            abstract=abstract,
        )
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("identifying protagonist")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        protagonist = self.get_chat_completion_content(messages=message_history)
        logger.info("protagonists: %s", protagonist)
        return protagonist

    def plan_story(self, abstract: str | None = None) -> dict:
        user_prompt_template = """Plan an interactive story in 5 chapters. {abstract_prompt_part}
        
You should loosely follow these bullet points:

1. Introduce the protagonist, {protagonist}, and their main goal, then the adventure starts
2. The protagonist encounter an obstacle
3. The protagonist finds some aid
4. The obstacle is cleared
5. The mission is complete and the story ends

The protagonist must be {protagonist}.
For each of these bullet points write a short sentence. Leave as many details as open as possible. They will be filled later depending on the reader's choice.
Format your output as a JSON dictionary, with integer numbers as keys. Do not output anything else."""

        if abstract:
            protagonist = self.identify_protagonist(abstract=abstract)
            abstract_prompt_part = f"The premise of the story is: {abstract}"
        else:
            protagonist = self.generate_protagonist()
            abstract_prompt_part = ""
        user_prompt = user_prompt_template.format(
            abstract_prompt_part=abstract_prompt_part,
            protagonist=protagonist,
        )
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("planning story")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("story plan: %s", model_response)
        story_plan = json_repair.loads(model_response)
        story_plan = {int(k): v for k, v in story_plan.items()}
        return story_plan

    def start_story(
        self,
        story_plan: dict,
    ) -> str:
        user_prompt_template = """# Story plan
{story_plan}

# Task
Write the first chapter of the story.
Introduce and describe the protagonist.
Set up a task or goal they need to do and add other details as you like.
Add a short paragraph describing how the protagonist sets off to achieve their goal.
Do NOT include elements that are part of the next chapters."""

        user_prompt = user_prompt_template.format(
            story_plan=json.dumps(story_plan, indent=4)
        )
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("starting story")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("story starts: %s", model_response)
        return model_response

    def describe_protagonist(self, story_so_far: str) -> str:
        user_prompt_template = """# The story so far:
{story_so_far}

# Task
Identify the protagonist of the story above.
Create a SHORT description (e.g. one paragraph) of the protagonist.
You must include the species, hair color, clothing or something special about their physical appearance.
If the protagonist is human, also include sex and age.
Do NOT add other details.
Do NOT include names.
Do NOT include actions or background, since these will be added in a second step.
Do NOT add comments or explanations"""

        user_prompt = user_prompt_template.format(story_so_far=story_so_far)
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("describing protagonist story")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("protagonist description: %s", model_response)
        return model_response

    def propose_continuation(
        self,
        story_so_far: str,
        n_alternatives: int = 2,
    ) -> list[str]:
        user_prompt_template = """# The story so far:
{story_so_far}

# Task:
Propose {n_alternatives} and only {n_alternatives} alternative ways in which the story could progress.
The alternatives must directly progress the story from its last sentence.

Keep each alternative as a short sentence of the form <PROTAGONIST><ACTION><DETAILS>.
Output the {n_alternatives} alternatives as a JSON dictionary, with numbers as keys and text as values (no list around the dictionary).
You MUST NOT output any comment, explanation or additional content."""

        user_prompt = user_prompt_template.format(
            story_so_far=story_so_far,
            n_alternatives=n_alternatives,
        )
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("proposing continuations")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("possible continuations: %s", model_response)
        alternatives_dict = json_repair.loads(model_response)
        alternatives = list(alternatives_dict.values())
        return alternatives

    def continue_story(
        self,
        story_plan: dict,
        story_so_far: str,
        chapter_nr: int,
        next_step: str,
    ) -> str:
        user_prompt_template = """# The story so far:
{story_so_far}

# Task:
Write chapter {chapter_nr} of the story.
Start from the following idea: {next_step}.
The chapter MUST conclude with the main event listed for chapter {chapter_nr} in the story plan: {chapter_end}.
The events of the chapter must unfold in a smooth way, without sudden changes or jumps from one scene to the next."""

        user_prompt = user_prompt_template.format(
            story_plan=json.dumps(story_plan, indent=4),
            story_so_far=story_so_far,
            chapter_nr=chapter_nr,
            next_step=next_step,
            chapter_end=story_plan[chapter_nr],
        )
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": user_prompt,
            },
        ]
        logger.info("Creating new chapter from idea %s", next_step)
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("story continues: %s", model_response)
        return model_response
