import json_repair
import openai

from config import local_config


class LLMStoryteller:

    client = openai.OpenAI(**local_config["openai_client_config"])
    system_prompt = """
* You are a storyteller for small children.
* You tell short instructive stories.
* You can set the stories in both realistic and fantastic worlds.
* You can use a few emojis to add visual elements.
* You MUST avoid mature themes like violence, guns, drugs, alcohol and sex.

* You will be asked to write chapters of a story.
* You MUST output one chapter at a time.
* Keep each chapter simple and short.
* After three of four chapters, you MUST start to close loose ends and converge to a conclusion.
* When you are ready to finish the story, add the following words "THE END".
* Otherwise do not add anything.

You only answer in {language}.
"""

    def __init__(self, language: str = "Italian") -> None:
        self.system_prompt = self.system_prompt.format(language=language)

    def start_story(self) -> str:
        start_story_prompt = """
* Start the story.
* Introduce and describe a main character.
* Set up a task or goal they need to do and add other details as you like.
* Do not progress the story too far.
"""
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": start_story_prompt,
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        return model_response

    def propose_alternatives(
        self,
        story_so_far: str,
        n_alternatives: int = 2,
    ) -> list[str]:
        propose_alternative_prompt = """
# The story so far:
{story_so_far}

# Task:
* Propose {n_alternatives} and only {n_alternatives} alternative ways in which the story could continue.
* Keep each alternative as a short sentence of the form <PROTAGONIST><ACTION><DETAILS>.
* After three of four chapters, you MUST propose alternatives
that complete the protagonist's mission and converge to a conclusion.
* Output the {n_alternatives} alternatives in JSON, with numbers as keys and text as values.
* You MUST NOT output any comment, explanation or additional content.
"""
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": propose_alternative_prompt.format(
                    story_so_far=story_so_far,
                    n_alternatives=n_alternatives,
                ),
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        print(model_response)
        alternatives_dict = json_repair.loads(model_response)
        alternatives = list(alternatives_dict.values())
        return alternatives

    def continue_story(self, story_so_far, next_step) -> str:
        continue_story_prompt = """
# The story so far:
{story_so_far}

# Task:
* Continue the story expanding on the following idea: {next_step}.
* Do not progress the story too far.
"""
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
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        return model_response

    def describe_protagonist(self, story_so_far: str) -> str:
        describe_character_prompt = """
# The story so far:
{story_so_far}

# Task
* Create a description of the protagonist.
* Include species, sex, age, eye color, hair color, clothing and distinctive traits.
* Do not include names.
* Do not include actions or background, since these will be added in a second step.
* Output the description as free text.

Just for this instruction, you MUST answer in English.
"""
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": describe_character_prompt.format(story_so_far=story_so_far),
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        return model_response
