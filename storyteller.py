import openai

from config import completion_params, openai_config


class LLMStoryteller:

    client = openai.OpenAI(**openai_config)
    system_prompt = """
You are a storyteller for small children.
You tell short instructive stories.
You can set the stories in both realistic and fantastic worlds.
You can use a few emojis to add visual elements.
You MUST avoid mature themes like violence, drugs, alcohol and sex.

You will be asked to write chapters of a story.
You MUST output one chapter at a time.
Keep each chapter simple and short.
After three of four chapters, you MUST start to close loose ends and converge to a conclusion.
When you are ready to finish the story, add the following words "THE END" to signal this is the last chapter, otherwise do not add anything.
You only answer in {language}.
"""

    def __init__(self, language: str = "Italian"):
        self.system_prompt = self.system_prompt.format(language=language)

    def start_story(self):
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": "Start the story. Introduce and describe a main character, set up a task or goal they need to do and add other details as you like. Do not progress the story too far.",
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **completion_params,
        )
        model_response = completion.choices[0].message.content
        return model_response

    def propose_alternatives(self, story_so_far):
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": f"# The story so far:\n{story_so_far}\n\n# Task:\nPropose three and only three alternative way in which the story could continue. Keep each alternative as a short sentence of the form <PROTAGONIST><ACTION><DETAILS>. After three of four chapters, you MUST propose alternatives that close loose ends and converge to a conclusion. Output the three alternatives on three different lines.",
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **completion_params,
        )
        model_response = completion.choices[0].message.content
        return model_response

    def continue_story(self, story_so_far, next_step):
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": f"# The story so far:\n{story_so_far}\n\n# Task:\nContinue the story expanding on the following idea: {next_step}. Do not progress the story too far.",
            },
        ]
        completion = self.client.chat.completions.create(
            messages=message_history,
            **completion_params,
        )
        model_response = completion.choices[0].message.content
        return model_response
