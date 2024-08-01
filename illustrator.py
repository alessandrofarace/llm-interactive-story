import openai

from config import local_config, openai_config


class ImageDescriber:
    client = openai.OpenAI(**local_config["openai_client_config"])
    system_prompt = """
Your task is to create a precise prompt for generating a picture with vision models.
Create a description as precise as possible, in free-form text.
"""

    def __init__(self):
        self.character_description = None

    def describe_character(self, story_so_far: str):
        describe_character_prompt = """
# The story so far:
{story_so_far}

# Task
Create a description of the protagonist.
Include species, sex, age, eye color, hair color, clothing and distinctive traits.
Do not include names.
Do not include actions or background, since these will be added in a second step.
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
        self.character_description = model_response

    def describe_picture(self, scene: str):
        describe_picture_prompt = """
# Protagonist:
{protagonist_description}

# Scene:
{scene}

# Task
Create a detailed prompt by combining the character and the scene.
Output the prompt in free text.
"""
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": describe_picture_prompt.format(
                    protagonist_description=self.character_description, scene=scene
                ),
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        return model_response
