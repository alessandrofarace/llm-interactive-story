import json
import logging

import json_repair

from config import config
from llm_utils import LLMAgent

logger = logging.getLogger(__name__)

storyteller_config = config["storyteller_config"]


# * You will be asked to write chapters of a story.
# * You MUST output one chapter at a time.
# * Keep each chapter simple and short.
# * After three of four chapters, you MUST start to close loose ends and converge to a conclusion.
# * When you are ready to finish the story, add the following words "THE END".
# * Otherwise do not add anything.

# You MUST output only the story, no comments like "here the story begins" or "here the chapter ends".


class LLMStoryteller(LLMAgent):

    config = storyteller_config
    system_prompt = """
You are a storyteller for small children.
You tell short instructive stories.
You can set the stories in both realistic and fantastic worlds.
You should mix a few emojis inbetween words to add visual elements.
You MUST avoid mature themes like violence, guns, drugs, alcohol and sex.

You stick to the instructions given and you NEVER add comments and additional explanations.
"""

    def plan_story(self) -> str:
        # TODO guide the generation of the protagonist
        user_prompt_template = """
Plan an interactive story in 6 chapters. You should loosely follow these bullet points:

1. Introduce a protagonist  (e.g. a boy or a girl, a speaking animal, or a fantastical creature) and their main goal
2. The adventure starts
3. The protagonist encounter an obstacle
4. The protagonist finds some aid
5. The obstacle is cleared
6. The mission is complete and the story ends

Be creative and original with the choice of the protagonist.
For each of these bullet points write a short sentence. Leave as many details as open as possible. They will be filled later depending on the reader's choice.
Format your output as a numbered list like the one above. Do not output anything else.
"""
        user_prompt = user_prompt_template
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
        return model_response

    def start_story(self, story_plan: str) -> str:
        user_prompt_template = """
# Story plan
{story_plan}

# Task
Write the first chapter of the story.
Introduce and describe the protagonist.
Set up a task or goal they need to do and add other details as you like.
Do not progress the story too far.
"""
        user_prompt = user_prompt_template.format(story_plan=story_plan)
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
        user_prompt_template = """
# The story so far:
{story_so_far}

# Task
Identify the protagonist of the story above.
Create a SHORT description (e.g. one paragraph) of the protagonist.
You must include the species, hair color, clothing or something special about their physical appearance.
If the protagonist is human, also include sex and age.
Do NOT add other details.
Do NOT include names.
Do NOT include actions or background, since these will be added in a second step.
Do NOT add comments or explanations
"""
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
        story_plan: str,
        story_so_far: str,
        chapter_nr: int,
        n_alternatives: int = 2,
    ) -> list[str]:
        user_prompt_template = """
# Story plan
{story_plan}

# The story so far:
{story_so_far}

# Task:
Propose {n_alternatives} and only {n_alternatives} alternative ways in which chapter {chapter_nr} could continue.
The alternatives MUST agree with the story plan.
Keep each alternative as a short sentence of the form <PROTAGONIST><ACTION><DETAILS>.
Output the {n_alternatives} alternatives as a JSON dictionary, with numbers as keys and text as values (no list around the dictionary).
You MUST NOT output any comment, explanation or additional content.
"""
        user_prompt = user_prompt_template.format(
            story_plan=story_plan,
            story_so_far=story_so_far,
            chapter_nr=chapter_nr,
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

    # ---

    def continue_story(self, story_so_far, next_step) -> str:
        continue_story_prompt = """
# The story so far:
{story_so_far}

# Task:
* Continue the story expanding on the following idea: {next_step}.
* Do not progress the story too far.
"""
        logger.info("Creating new chapter from idea %s", next_step)
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": continue_story_prompt.format(
                    story_so_far=story_so_far, next_step=next_step
                ),
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **storyteller_config["chat_completion_params"],
        )
        model_response = completion.choices[0].message.content
        logger.info("story continues: %s", model_response)
        return model_response
