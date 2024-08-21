"""
Chat app and UI
"""

import logging
import shutil
from base64 import b64decode

import streamlit as st

from config import config
from illustrator import SDXLLightningLocalIllustrator
from story import Chapter, OpenEndedStory
from storyteller import LLMStoryteller
from translator import LLMTranslator


@st.cache_resource
def configure_logger():
    logger = logging.getLogger()
    log_file = "log_file.log"
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = configure_logger()

N_ALTERNATIVES = config["storyteller_config"]["n_alternatives"]


def start_story() -> None:
    st.session_state.story = OpenEndedStory()
    st.session_state.storyteller = LLMStoryteller()
    st.session_state.illustrator = SDXLLightningLocalIllustrator()
    st.session_state.translator = LLMTranslator(language=st.session_state.language)

    st.session_state.story.story_plan = st.session_state.storyteller.plan_story()

    story_start = st.session_state.storyteller.start_story(
        story_plan=st.session_state.story.story_plan
    )
    story_start_translated = st.session_state.translator.translate(story_start)
    st.session_state.story.add_chapter(
        text=story_start,
        translated_text=story_start_translated,
    )

    st.session_state.story.protagonist_description = (
        st.session_state.storyteller.describe_protagonist(
            story_so_far=st.session_state.story.full_text
        )
    )
    protagonist_image = st.session_state.illustrator.create_picture(
        protagonist_description=st.session_state.story.protagonist_description,
        scene="No scene has been set, just draw the full-body protagonist",
    )
    image_data = b64decode(protagonist_image)
    image_path = "images/protagonist.png"
    with open(image_path, "wb") as png:
        png.write(image_data)
    st.session_state.story.chapters[0].image = image_path

    possible_continuations = st.session_state.storyteller.propose_continuation(
        story_plan=st.session_state.story.story_plan,
        story_so_far=st.session_state.story.full_text,
        chapter_nr=len(st.session_state.story.chapters),
        n_alternatives=N_ALTERNATIVES,
    )
    st.session_state.story.reset_continuations()
    for j, continuation_text in enumerate(possible_continuations):
        continuation_image = st.session_state.illustrator.create_picture(
            protagonist_description=st.session_state.story.protagonist_description,
            scene=continuation_text,
        )
        image_data = b64decode(continuation_image)
        image_path = f"images/alt_{j}.png"
        with open(image_path, "wb") as png:
            png.write(image_data)
        continuation_text_translated = st.session_state.translator.translate(
            continuation_text
        )
        st.session_state.story.add_continuation(
            text=continuation_text,
            image=image_path,
            translated_text=continuation_text_translated,
        )

    st.rerun()


def write_chapter(continuation: Chapter) -> None:
    next_step_text = continuation.text
    n_chapters = len(st.session_state.story.chapters)
    next_step_image = f"images/chap_{n_chapters}.png"
    shutil.copy(continuation.image, next_step_image)
    chapter = st.session_state.storyteller.continue_story(
        st.session_state.story.full_text,
        next_step_text,
    )
    chapter_translated = st.session_state.translator.translate(chapter)
    st.session_state.story.add_chapter(
        text=chapter,
        image=next_step_image,
        translated_text=chapter_translated,
    )
    possible_continuations = st.session_state.storyteller.propose_continuation(
        story_so_far=st.session_state.story.full_text,
        n_alternatives=N_ALTERNATIVES,
    )
    st.session_state.story.reset_continuations()
    for j, continuation_text in enumerate(possible_continuations):
        continuation_image = st.session_state.illustrator.create_picture(
            protagonist_description=st.session_state.story.protagonist_description,
            scene=continuation_text,
        )
        image_data = b64decode(continuation_image)
        image_path = f"images/alt_{j}.png"
        with open(image_path, "wb") as png:
            png.write(image_data)
        continuation_text_translated = st.session_state.translator.translate(
            continuation_text
        )
        st.session_state.story.add_continuation(
            text=continuation_text,
            image=image_path,
            translated_text=continuation_text_translated,
        )


def display_chapter(chapter: Chapter) -> None:
    if chapter.image:
        st.image(
            chapter.image,
            width=512,
        )
    st.markdown(chapter.translated_text)


def display_continuation(continuation: Chapter, col_pos: int) -> None:
    st.image(
        continuation.image,
        continuation.translated_text,
        width=256,
    )
    st.button(
        "Continue story",
        key=f"write_chapter_{str(col_pos)}",
        on_click=write_chapter,
        args=[continuation],
    )


if "story" not in st.session_state:
    st.session_state.story = None
if "language" not in st.session_state:
    st.session_state.language = None
if "storyteller" not in st.session_state:
    st.session_state.storyteller = None
if "illustrator" not in st.session_state:
    st.session_state.illustrator = None
if "translator" not in st.session_state:
    st.session_state.translator = None


if st.session_state.language is None:
    language = st.selectbox("Language", ("Italian", "German", "English"), index=None)
    if language is not None:
        st.session_state.language = language
        start_story()
else:
    if st.session_state.story:
        for chapter in st.session_state.story.chapters:
            display_chapter(chapter)

        if st.session_state.story.possible_continuations:
            n_continuations = len(st.session_state.story.possible_continuations)
            cols = st.columns([1] * n_continuations)
            for continuation, col, col_pos in zip(
                st.session_state.story.possible_continuations,
                cols,
                list(range(n_continuations)),
            ):
                with col:
                    display_continuation(continuation=continuation, col_pos=col_pos)
