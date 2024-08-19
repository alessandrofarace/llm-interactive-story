import base64
import logging
import random
from base64 import b64encode
from io import BytesIO

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


class LocalDiffusersIllustrator:

    style = None
    width = None
    height = None
    n_steps = None

    def __init__(self):
        self.seed = random.randint(1, 1000000)

    def describe_picture(self, protagonist_description, scene):
        raise NotImplementedError

    def get_pipeline(self):
        raise NotImplementedError

    def create_picture(self, protagonist_description: str, scene: str) -> str:

        import torch

        device = "mps"

        inp = {
            "short_name": "img",
            "prompt": self.describe_picture(
                protagonist_description=protagonist_description,
                scene=scene,
            )
            + f", {self.style}",
            "width": self.width,
            "height": self.height,
            "n_steps": self.n_steps,
            "seed": self.seed,
        }

        pipe = self.get_pipeline()
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
        del pipe
        return img_str


class SDXLLightningLocalIllustrator(LocalDiffusersIllustrator):
    llm_client = openai.OpenAI(**local_config["openai_client_config"])
    system_prompt = """
Your task is to create a precise prompt for generating a picture with stable diffusion xl.
Create a description as precise as possible, in the best format for stable diffusion xl.
You MUST include a few physical descriptors and a short description of the scene.
Output the prompt as a a comma separated list of minimal sentences.
Limit the output length to around 50 tokens.
You MUST follow the instructions given, without adding any unrequired information.
"""
    describe_picture_prompt = """
# Protagonist:
{protagonist_description}

# Scene:
{scene}

# Task
Create a schematic prompt by combining the character and the scene.
Use only the information provided above. You can add atmosphere attributes, but do NOT invent other elements.

You MUST ouptut only the prompt. Do NOT add any comment or explanation.
"""

    style = "digital painting, cartoon, pastel colors, best quality"
    width = 1024
    height = 1024
    n_steps = 4

    def describe_picture(self, protagonist_description: str, scene: str) -> str:
        message_history = [
            {"role": "system", "content": self.system_prompt},
            {
                "name": "user",
                "role": "user",
                "content": self.describe_picture_prompt.format(
                    protagonist_description=protagonist_description,
                    scene=scene,
                ),
            },
        ]
        message_history_with_examples = (
            message_history[0:1] + self._get_examples() + message_history[-1:]
        )
        completion = self.llm_client.chat.completions.create(
            messages=message_history_with_examples,
            **local_config["completion_params"],
        )
        model_response = completion.choices[0].message.content
        logger.info("prompt for image generation: %s", model_response)
        return model_response

    def get_pipeline(self):
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download

        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_lora.safetensors"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        return pipe

    def _get_examples(self) -> list[dict]:
        examples = [
            {
                "protagonist_description": "Female rabbit, 6 months old, bright brown eyes, fluffy white fur, wearing a tiny backpack with a small, colorful scarf tied around her neck",
                "scene": "No scene has been set, just draw the full-body protagonist",
                "image_prompt": "A female rabbit, bright brown eyes, fluffy white fur, tiny backpack, small colorful scarf tied around neck, full-body, front view",
            },
            {
                "protagonist_description": "Female rabbit, 6 months old, bright brown eyes, fluffy white fur, wearing a tiny backpack with a small, colorful scarf tied around her neck",
                "scene": "Rosie encounters a friendly squirrel named Squeaky who offers to guide her through the forest and share his knowledge of the Spring's location.",
                "image_prompt": "A female rabbit, bright brown eyes, fluffy white fur, tiny backpack, colorful scarf, speaking with a friendly squirrel, warm sunlight, tall trees, soft forest floor",
            },
        ]
        example_messages = []
        for example in examples:
            example_messages.append(
                {
                    "name": "user",
                    "role": "user",
                    "content": self.describe_picture_prompt.format(
                        protagonist_description=example["protagonist_description"],
                        scene=example["scene"],
                    ),
                }
            )
            example_messages.append(
                {
                    "name": "assistant",
                    "role": "assistant",
                    "content": example["image_prompt"],
                }
            )
        return example_messages


class SD15HyperLocalIllustrator(LocalDiffusersIllustrator):
    llm_client = openai.OpenAI(**local_config["openai_client_config"])
    system_prompt = """
Your task is to create a precise prompt for generating a picture with stable diffusion 1.5.
Create a description as precise as possible, in the best format for stable diffusion 1.5.
You MUST include a few physical descriptors and a short description of the scene.
Output the prompt as a a comma separated list of attributes and objects, no verbs or grammar is needed.
Limit the output length to around 50 tokens.
You MUST follow the instructions given, without adding any unrequired information.
"""
    style = "digital painting, cartoon, pastel colors, best quality"
    width = 512
    height = 512
    n_steps = 8

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

    def get_pipeline(self):
        import torch
        from diffusers import DDIMScheduler, DiffusionPipeline
        from huggingface_hub import hf_hub_download

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
        return pipe


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
