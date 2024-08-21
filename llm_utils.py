import openai


class LLMAgent:

    config = {}
    system_prompt = None

    def __init__(self) -> None:
        self.client = openai.OpenAI(**self.config["openai_client_config"])

    def get_chat_completion_content(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            messages=messages,
            **self.config["chat_completion_params"],
        )
        model_response = completion.choices[0].message.content
        return model_response


def user_message(content):
    return {
        "name": "user",
        "role": "user",
        "content": content,
    }


def assistant_message(content):
    return {
        "name": "assistant",
        "role": "assistant",
        "content": content,
    }
