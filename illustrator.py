import logging
from base64 import b64encode

import openai

from config import local_config, openai_config

logger = logging.getLogger(__name__)


class MockIllustrator:
    def create_picture(self, protagonist_description: str, scene: str) -> str:
        image_path = (
            "/Users/al.farace/Projects/llm-interactive-story/images/protagonist.png"
        )
        with open(image_path, "rb") as png:
            image_data = png.read()
        return b64encode(image_data)


class SD15HyperLocalIllustrator:
    llm_client = openai.OpenAI(**local_config["openai_client_config"])
    system_prompt = """
Your task is to create a precise prompt for generating a picture with stable diffusion 1.5.
Create a description as precise as possible, in the best format for stable diffusion 1.5.
You MUST include EXACTLY 4 physical descriptors and a short description of the scene, all comma separated.
Limit the output length to around 50 tokens.
You MUST follow the instructions given, without adding any unrequired information.
"""
    style = "watercolor, pastel colors, best quality"

    def describe_picture(self, protagonist_description: str, scene: str) -> str:
        describe_picture_prompt = """
# Protagonist:
{protagonist_description}

# Scene:
{scene}

# Task
Create a schematic prompt by combining the character and the scene.

You MUST ouptut only the prompt. Do NOT add any comment or explanation.
"""
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": describe_picture_prompt.format(
                    protagonist_description=protagonist_description,
                    scene=scene,
                ),
            },
        ]
        completion = self.llm_client.chat.completions.create(
            messages=message_history,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        logger.info("prompt for image generation: %s", model_response)
        return model_response

    def create_picture(self, protagonist_description: str, scene: str) -> str:
        import base64
        import random
        from io import BytesIO

        import torch
        from diffusers import DDIMScheduler, DiffusionPipeline
        from huggingface_hub import hf_hub_download

        device = "mps"

        inp = {
            "short_name": "img",
            "prompt": self.describe_picture(
                protagonist_description=protagonist_description,
                scene=scene,
            )
            + f", {self.style}",
            "width": 512,
            "height": 512,
            "n_steps": 8,
            "seed": random.randint(1, 1000000),
        }

        base_model_id = "runwayml/stable-diffusion-v1-5"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SD15-8steps-lora.safetensors"
        pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora()
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        pipe.to(device)

        generator = [torch.Generator(device).manual_seed(inp["seed"])]

        image = pipe(
            prompt=inp["prompt"],
            generator=generator,
            num_inference_steps=inp["n_steps"],
            guidance_scale=0.2,
            width=inp["width"],
            height=inp["height"],
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str


class Dalle3Illustrator:
    llm_client = openai.OpenAI(**local_config["openai_client_config"])
    vision_client = openai.OpenAI(**openai_config["openai_client_config"])
    system_prompt = """
Your task is to create a precise prompt for generating a picture with dall-e 3.
Create a description as precise as possible, in the best format for dall-e 3.
You MUST follow the instructions given, without adding any unrequired information.
"""

    #     def describe_character(self, story_so_far: str):
    #         describe_character_prompt = """
    # # The story so far:
    # {story_so_far}

    # # Task
    # Create a description of the protagonist.
    # Include species, sex, age, eye color, hair color, clothing and distinctive traits.
    # Do not include names.
    # Do not include actions or background, since these will be added in a second step.
    # """
    #         message_history = [
    #             {"role": "system", "content": self.system_prompt},
    #             {
    #                 "name": "user",
    #                 "role": "user",
    #                 "content": describe_character_prompt.format(story_so_far=story_so_far),
    #             },
    #         ]
    #         completion = self.client.chat.completions.create(
    #             messages=message_history,
    #             **local_config["completion_params"],
    #         )
    #         model_response = completion.choices[0].message.content
    #         self.character_description = model_response

    def describe_picture(self, protagonist_description: str, scene: str) -> str:
        describe_picture_prompt = """
# Protagonist:
{protagonist_description}

# Scene:
{scene}

# Task
Create a detailed prompt by combining the character and the scene.
Output the prompt in in the best format for dall-e 3.

You MUST ouptut only the prompt. Do NOT add any comment or explanation.
"""
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": describe_picture_prompt.format(
                    protagonist_description=protagonist_description,
                    scene=scene,
                ),
            },
        ]
        completion = self.llm_client.chat.completions.create(
            messages=message_history,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        return model_response

    def create_picture(self, protagonist_description: str, scene: str) -> str:
        prompt = self.describe_picture(
            protagonist_description=protagonist_description,
            scene=scene,
        )
        print("---" * 20)
        print("Protagonist description image prompt:")
        print(prompt)
        response = self.vision_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json",
        )

        image_json = response.data[0].b64_json
        return image_json


# class ImageDescriber:
#             client = openai.OpenAI(**local_config["openai_client_config"])
#             system_prompt = """
#         Your task is to create a precise prompt for generating a picture with vision models.
#         Create a description as precise as possible, in free-form text.
#         """

#             def __init__(self):
#                 self.character_description = None

#             def describe_character(self, story_so_far: str):
#                 describe_character_prompt = """
#         # The story so far:
#         {story_so_far}

#         # Task
#         Create a description of the protagonist.
#         Include species, sex, age, eye color, hair color, clothing and distinctive traits.
#         Do not include names.
#         Do not include actions or background, since these will be added in a second step.
#         """
#                 message_history = [
#                     {"role": "system", "content": self.system_prompt},
#                     {
#                         "name": "user",
#                         "role": "user",
#                         "content": describe_character_prompt.format(story_so_far=story_so_far),
#                     },
#                 ]
#                 completion = self.client.chat.completions.create(
#                     messages=message_history,
#                     **local_config["completion_params"],
#                 )
#                 model_response = completion.choices[0].message.content
#                 self.character_description = model_response

#             def describe_picture(self, scene: str):
#         describe_picture_prompt = """
# # Protagonist:
# {protagonist_description}

# # Scene:
# {scene}

# # Task
# Create a detailed prompt by combining the character and the scene.
# Output the prompt in free text.
# """
#         message_history = [
#             {"role": "system", "content": self.system_prompt},
#             {
#                 "name": "user",
#                 "role": "user",
#                 "content": describe_picture_prompt.format(
#                     protagonist_description=self.character_description, scene=scene
#                 ),
#             },
#         ]
#         completion = self.client.chat.completions.create(
#             messages=message_history,
#             **local_config["completion_params"],
#         )
#         model_response = completion.choices[0].message.content
#         return model_response
