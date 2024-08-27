import base64
import json
import logging
import random
from base64 import b64encode
from io import BytesIO

import json_repair
import openai
import requests
from gradio_client import Client

from config import config
from llm_utils import LLMAgent, assistant_message, user_message

logger = logging.getLogger(__name__)

illustrator_config = config["illustrator_config"]


class LLMPrompterForSDXL(LLMAgent):
    config = illustrator_config
    system_prompt = """
You are a gen-ai artist expert in prompting stable diffusion xl (SD-XL) models.
Output the requested prompts as a a comma separated list of minimal sentences.
"""

    def create_attributes_protagonist_and_atmosphere(
        self, protagonist_description: str
    ):
        system_prompt = (
            self.system_prompt
            + """# Task
Create 4 or 5 short sentences that describe the protagonist, including the species. You MUST include a few physical descriptors. Do NOT focus too much on the face.
Create 2 or 3 attributes that describe the general atmosphere (e.g. relaxing, adventurous, mysterious, ...)
Output the attributes as a comma separated list of minimal sentences.

Do NOT add any comment or explanation. You MUST follow the instructions given, without adding any unrequired information.
Format the output as a json dict

Example:
{{
    "protagonist": ["A male squirrel", "brown fur", "blue tail", "blue pointy hat", "bowtie around his neck"],
    "atmosphere": ["foggy", "cold", "mysterious"]
}}"""
        )
        user_prompt_template = """# Protagonist:
{protagonist_description}"""
        user_prompt = user_prompt_template.format(
            protagonist_description=protagonist_description
        )
        message_history = [
            {"role": "system", "content": system_prompt},
            user_message(
                user_prompt_template.format(
                    protagonist_description="She is a young Luminari, with hair as bright as the stars and eyes that shine like the morning dew. Her species is characterized by her sparkling appearance, and her hair is a notable feature, being a vibrant, star-like color."
                )
            ),
            assistant_message(
                '{\n  "protagonist": ["a young female luminari", "hair as bright as the stars", "sparkling blue dress"],\n "atmosphere": ["magical", "enchanting", "whimsical"]\n}'
            ),
            user_message(
                user_prompt_template.format(
                    protagonist_description="This young shape-shifting fox has fur that shimmers like the stars on a clear night and eyes that sparkle with curiosity. She has no discernible clothing, but her fur is her natural, shimmering coat."
                )
            ),
            assistant_message(
                '{\n  "protagonist": ["a young shape-shifting fox", "fur that shimmers like the stars"],\n "atmosphere": ["adventurous", "mysterious", "enchanted"]\n}'
            ),
            user_message(user_prompt),
        ]
        logger.info("creating image attributes for protagonist and atmosphere")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("protagonist and atmosphere description: %s", model_response)
        parsed_response = json_repair.loads(model_response)
        protagonist_attributes = ", ".join(parsed_response["protagonist"])
        atmosphere_attributes = ", ".join(parsed_response["atmosphere"])
        return protagonist_attributes, atmosphere_attributes

    def create_scene_attributes(self, scene: str) -> str:
        system_prompt = (
            self.system_prompt
            + """# Task
Create a schematic prompt that describes the scene.
Use only the information provided above, do NOT invent other elements.
Output 2 or 3 short sentences, separated by a commma.
Do NOT describe the protagonist, only other elements in the scene.

If there is no scene, ouptut the following "full-body, neutral background".

Do NOT add any comment or explanation. You MUST follow the instructions given, without adding any unrequired information."""
        )
        user_prompt_template = """# Scene:
{scene}"""
        user_prompt = user_prompt_template.format(
            scene=scene,
        )
        message_history = [
            {"role": "system", "content": system_prompt},
            user_message(
                user_prompt_template.format(
                    scene="No scene has been set, just draw the full-body protagonist"
                )
            ),
            assistant_message("full-body, neutral background"),
            user_message(
                user_prompt_template.format(
                    scene="Bella flutters hesitantly, landing on a leaf that holds a secret message."
                )
            ),
            assistant_message("on a big leaf, secret message"),
            user_message(
                user_prompt_template.format(
                    scene="Luna swims through a kelp forest, where she meets a wise old sea turtle who gives her a cryptic map to find the Pearl of the Sea."
                )
            ),
            assistant_message("kelp forest, old sea turtle, treasure map"),
            user_message(
                user_prompt_template.format(
                    scene="Luna explores a sunken ship, where she discovers a hidden compartment containing a mysterious compass that might lead her to the Pearl"
                )
            ),
            assistant_message("sunken ship, hidden mysterious compass"),
            user_message(
                user_prompt_template.format(
                    scene="Luna stumbles upon a group of mischievous fairies playing tricks on each other near a sparkling waterfall"
                )
            ),
            assistant_message("group of fairies, sparkling waterfall"),
            user_message(user_prompt),
        ]
        logger.info("creating image attributes for the scene")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("scene description: %s", model_response)
        return model_response


class LLMPrompterForFlux(LLMAgent):
    config = illustrator_config
    system_prompt = """
You are a gen-ai artist expert in prompting diffusion Flux models.
Output the requested prompts as short descriptive sentences.
"""

    def describe_picture(
        self,
        protagonist_description: str,
        scene: str,
    ) -> str:
        system_prompt = (
            self.system_prompt
            + """# Task
You will be asked to write a short prompt that describes a scene.
Use only the information provided by the user, do NOT invent other elements.
Output a few short sentences, separated by a commma.
Roughly follow this schema: protagonist description, main actions of the protagonist,other character and elements in the scene, atmosphere of the picture

Do NOT add any comment or explanation. You MUST follow the instructions given, without adding any unrequired information."""
        )
        user_prompt_template = """# Protagonist description:
{protagonist_description}

# Scene:
{scene}"""
        user_prompt = user_prompt_template.format(
            protagonist_description=protagonist_description,
            scene=scene,
        )
        message_history = [
            {"role": "system", "content": system_prompt},
            user_message(user_prompt),
        ]
        logger.info("creating image prompt")
        logger.info("messages: %s", json.dumps(message_history, indent=4))
        model_response = self.get_chat_completion_content(messages=message_history)
        logger.info("image prompt: %s", model_response)
        return model_response


class LocalDiffusersIllustrator:

    style_attributes = None
    width = None
    height = None
    n_steps = None

    def __init__(self):
        self.seed = random.randint(1, 1000000)

    def get_pipeline(self):
        raise NotImplementedError

    def create_image_prompt(self, protagonist_description: str, scene: str) -> str:
        raise NotImplementedError

    def create_b64_encoded_picture(
        self, protagonist_description: str, scene: str
    ) -> str:

        import torch

        device = "mps"

        inp = {
            "short_name": "img",
            "prompt": self.create_image_prompt(
                protagonist_description=protagonist_description,
                scene=scene,
            ),
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
    style_attributes = "digital painting, cartoon, pastel colors, best quality"
    width = 1024
    height = 1024
    n_steps = 4

    def __init__(self):
        super().__init__()
        self.protagonist_attributes = None
        self.atmosphere_attributes = None
        self.llm_prompter = LLMPrompterForSDXL()

    def set_protagonist_and_atmosphere_attributes(
        self,
        protagonist_description: str,
    ) -> None:
        self.protagonist_attributes, self.atmosphere_attributes = (
            self.llm_prompter.create_attributes_protagonist_and_atmosphere(
                protagonist_description=protagonist_description
            )
        )

    def create_image_prompt(self, protagonist_description: str, scene: str) -> str:
        if self.protagonist_attributes is None:
            self.set_protagonist_and_atmosphere_attributes(
                protagonist_description=protagonist_description
            )
        scene_attributes = self.llm_prompter.create_scene_attributes(scene=scene)
        image_prompt = ", ".join(
            [
                self.protagonist_attributes,
                scene_attributes,
                self.atmosphere_attributes,
                self.style_attributes,
            ]
        )
        logger.info("image_prompt: %s", image_prompt)
        return image_prompt

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

    # def _get_examples(self) -> list[dict]:
    #     examples = [
    #         {
    #             "protagonist_description": "Female rabbit, 6 months old, bright brown eyes, fluffy white fur, wearing a tiny backpack with a small, colorful scarf tied around her neck",
    #             "scene": "No scene has been set, just draw the full-body protagonist",
    #             "image_prompt": "A female rabbit, bright brown eyes, fluffy white fur, tiny backpack, small colorful scarf tied around neck, full-body, front view",
    #         },
    #         {
    #             "protagonist_description": "Female rabbit, 6 months old, bright brown eyes, fluffy white fur, wearing a tiny backpack with a small, colorful scarf tied around her neck",
    #             "scene": "Rosie encounters a friendly squirrel named Squeaky who offers to guide her through the forest and share his knowledge of the Spring's location.",
    #             "image_prompt": "A female rabbit, bright brown eyes, fluffy white fur, tiny backpack, colorful scarf, speaking with a friendly squirrel, warm sunlight, tall trees, soft forest floor",
    #         },
    #     ]
    #     example_messages = []
    #     for example in examples:
    #         example_messages.append(
    #             {
    #                 "name": "user",
    #                 "role": "user",
    #                 "content": self.describe_picture_prompt.format(
    #                     protagonist_description=example["protagonist_description"],
    #                     scene=example["scene"],
    #                 ),
    #             }
    #         )
    #         example_messages.append(
    #             {
    #                 "name": "assistant",
    #                 "role": "assistant",
    #                 "content": example["image_prompt"],
    #             }
    #         )
    #     return example_messages


class FluxClientIllustrator:
    style_attributes = "digital painting, cartoon, pastel colors, best quality"
    width = 1024
    height = 1024
    n_steps = 4

    def __init__(self):
        self.llm_prompter = LLMPrompterForFlux()
        self.client = Client(
            "black-forest-labs/FLUX.1-schnell",
            download_files=False,
        )
        self.seed = random.randint(1, 1000000)

    def create_image_prompt(self, protagonist_description: str, scene: str) -> str:
        image_prompt = self.llm_prompter.describe_picture(
            protagonist_description=protagonist_description,
            scene=scene,
        )
        image_prompt = f"{image_prompt}, {self.style_attributes}"
        logger.info("image_prompt: %s", image_prompt)
        return image_prompt

    def create_b64_encoded_picture(
        self,
        protagonist_description: str,
        scene: str,
    ) -> str:
        inp = {
            "short_name": "img",
            "prompt": self.create_image_prompt(
                protagonist_description=protagonist_description,
                scene=scene,
            ),
            "width": self.width,
            "height": self.height,
            "n_steps": self.n_steps,
            "seed": self.seed,
        }

        result = self.client.predict(
            prompt=inp["prompt"],
            seed=self.seed,
            randomize_seed=False,
            width=inp["width"],
            height=inp["height"],
            num_inference_steps=inp["n_steps"],
            api_name="/infer",
        )
        r = requests.get(result[0]["url"])
        buffered = BytesIO(r.content)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str


class MockIllustrator:
    def create_b64_encoded_picture(
        self,
        protagonist_description: str,
        scene: str,
    ) -> str:
        image_path = (
            "/Users/al.farace/Projects/llm-interactive-story/images/protagonist.png"
        )
        with open(image_path, "rb") as png:
            image_data = png.read()
        return b64encode(image_data)


# class SD15HyperLocalIllustrator(LocalDiffusersIllustrator):
#     llm_client = openai.OpenAI(**illustrator_config["llm_openai_client_config"])
#     system_prompt = """
# Your task is to create a precise prompt for generating a picture with stable diffusion 1.5.
# Create a description as precise as possible, in the best format for stable diffusion 1.5.
# You MUST include a few physical descriptors and a short description of the scene.
# Output the prompt as a a comma separated list of attributes and objects, no verbs or grammar is needed.
# Limit the output length to around 50 tokens.
# You MUST follow the instructions given, without adding any unrequired information.
# """
#     style = "digital painting, cartoon, pastel colors, best quality"
#     width = 512
#     height = 512
#     n_steps = 8

#     def describe_picture(self, protagonist_description: str, scene: str) -> str:
#         describe_picture_prompt = """
# # Protagonist:
# {protagonist_description}

# # Scene:
# {scene}

# # Task
# Create a schematic prompt by combining the character and the scene.

# You MUST ouptut only the prompt. Do NOT add any comment or explanation.
# """
#         message_history = [
#             {"role": "system", "content": self.system_prompt},
#             {
#                 "name": "user",
#                 "role": "user",
#                 "content": describe_picture_prompt.format(
#                     protagonist_description=protagonist_description,
#                     scene=scene,
#                 ),
#             },
#         ]
#         completion = self.llm_client.chat.completions.create(
#             messages=message_history,
#             **illustrator_config["chat_completion_params"],
#         )
#         model_response = completion.choices[0].message.content
#         logger.info("prompt for image generation: %s", model_response)
#         return model_response

#     def get_pipeline(self):
#         import torch
#         from diffusers import DDIMScheduler, DiffusionPipeline
#         from huggingface_hub import hf_hub_download

#         base_model_id = "runwayml/stable-diffusion-v1-5"
#         repo_name = "ByteDance/Hyper-SD"
#         ckpt_name = "Hyper-SD15-8steps-lora.safetensors"
#         pipe = DiffusionPipeline.from_pretrained(
#             base_model_id,
#             torch_dtype=torch.float16,
#             variant="fp16",
#         )
#         pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
#         pipe.fuse_lora()
#         pipe.scheduler = DDIMScheduler.from_config(
#             pipe.scheduler.config,
#             timestep_spacing="trailing",
#         )
#         return pipe


# class Dalle3Illustrator:
#     llm_client = openai.OpenAI(**illustrator_config["llm_openai_client_config"])
#     vision_client = openai.OpenAI(**illustrator_config["vision_openai_client_config"])
#     system_prompt = """
# Your task is to create a precise prompt for generating a picture with dall-e 3.
# Create a description as precise as possible, in the best format for dall-e 3.
# You MUST follow the instructions given, without adding any unrequired information.
# """

#     #     def describe_character(self, story_so_far: str):
#     #         describe_character_prompt = """
#     # # The story so far:
#     # {story_so_far}

#     # # Task
#     # Create a description of the protagonist.
#     # Include species, sex, age, eye color, hair color, clothing and distinctive traits.
#     # Do not include names.
#     # Do not include actions or background, since these will be added in a second step.
#     # """
#     #         message_history = [
#     #             {"role": "system", "content": self.system_prompt},
#     #             {
#     #                 "name": "user",
#     #                 "role": "user",
#     #                 "content": describe_character_prompt.format(story_so_far=story_so_far),
#     #             },
#     #         ]
#     #         completion = self.client.chat.completions.create(
#     #             messages=message_history,
#     #             **illustrator_config["chat_completion_params"],
#     #         )
#     #         model_response = completion.choices[0].message.content
#     #         self.character_description = model_response

#     def describe_picture(self, protagonist_description: str, scene: str) -> str:
#         describe_picture_prompt = """
# # Protagonist:
# {protagonist_description}

# # Scene:
# {scene}

# # Task
# Create a detailed prompt by combining the character and the scene.
# Output the prompt in in the best format for dall-e 3.

# You MUST ouptut only the prompt. Do NOT add any comment or explanation.
# """
#         message_history = [
#             {"role": "system", "content": self.system_prompt},
#             {
#                 "name": "user",
#                 "role": "user",
#                 "content": describe_picture_prompt.format(
#                     protagonist_description=protagonist_description,
#                     scene=scene,
#                 ),
#             },
#         ]
#         completion = self.llm_client.chat.completions.create(
#             messages=message_history,
#             **illustrator_config["chat_completion_params"],
#         )
#         model_response = completion.choices[0].message.content
#         return model_response

#     def create_picture(self, protagonist_description: str, scene: str) -> str:
#         prompt = self.describe_picture(
#             protagonist_description=protagonist_description,
#             scene=scene,
#         )
#         print("---" * 20)
#         print("Protagonist description image prompt:")
#         print(prompt)
#         response = self.vision_client.images.generate(
#             model="dall-e-3",
#             prompt=prompt,
#             size="1024x1024",
#             quality="standard",
#             n=1,
#             response_format="b64_json",
#         )

#         image_json = response.data[0].b64_json
#         return image_json
