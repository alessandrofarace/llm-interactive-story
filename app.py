"""
Chat app and UI
"""

import json
import logging
import os
import shutil
from base64 import b64decode

import streamlit as st

from config import story_config
from illustrator import (
    FluxClientIllustrator,
    MockIllustrator,
    SDXLLightningLocalIllustrator,
)
from story import Chapter, OpenEndedStory
from storyteller import LLMStoryteller
from translator import LLMTranslator


@st.cache_resource
def configure_logger():
    logger = logging.getLogger()
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/log_file.log"
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = configure_logger()

N_ALTERNATIVES = story_config.n_continuations


def add_continuations() -> None:
    possible_continuations = st.session_state.storyteller.propose_continuation(
        story_so_far=st.session_state.story.full_text,
        n_alternatives=N_ALTERNATIVES,
    )
    st.session_state.story.reset_continuations()
    for j, continuation_text in enumerate(possible_continuations):
        continuation_image = st.session_state.illustrator.create_b64_encoded_picture(
            protagonist_description=st.session_state.story.protagonist_description,
            scene=continuation_text,
        )
        image_data = b64decode(continuation_image)
        image_path = f"images/alt_{j}.png"
        with open(image_path, "wb") as png:
            png.write(image_data)
        continuation_text_translated = (
            st.session_state.translator.translate_to_language(continuation_text)
        )
        st.session_state.story.add_continuation(
            text=continuation_text,
            image=image_path,
            translated_text=continuation_text_translated,
        )


def start_story(language: str | None, user_input: str | None) -> None:
    if not language:
        st.error("Please select a language")
        return

    st.session_state.language = language
    st.session_state.storyteller = LLMStoryteller()
    st.session_state.illustrator = FluxClientIllustrator()
    st.session_state.translator = LLMTranslator(language=st.session_state.language)

    if user_input != "Suprise me":
        abstract_english = st.session_state.translator.translate_to_english(user_input)
    else:
        abstract_english = None

    st.session_state.story = OpenEndedStory(abstract=abstract_english)
    st.session_state.started = True

    st.session_state.story.story_plan = st.session_state.storyteller.plan_story(
        abstract=st.session_state.story.abstract
    )

    story_start = st.session_state.storyteller.start_story(
        story_plan=st.session_state.story.story_plan
    )
    story_start_translated = st.session_state.translator.translate_to_language(
        story_start
    )
    st.session_state.story.add_chapter(
        text=story_start,
        translated_text=story_start_translated,
    )

    st.session_state.story.protagonist_description = (
        st.session_state.storyteller.describe_protagonist(
            story_so_far=st.session_state.story.full_text
        )
    )
    protagonist_image = st.session_state.illustrator.create_b64_encoded_picture(
        protagonist_description=st.session_state.story.protagonist_description,
        scene="No scene has been set, just draw the full-body protagonist",
    )
    image_data = b64decode(protagonist_image)
    image_path = "images/protagonist.png"
    with open(image_path, "wb") as png:
        png.write(image_data)
    st.session_state.story.chapters[0].image = image_path

    add_continuations()

    # st.rerun()


def write_chapter(continuation: Chapter) -> None:
    next_step_text = continuation.text
    n_chapters = len(st.session_state.story.chapters)
    next_step_image = f"images/chap_{n_chapters}.png"
    shutil.copy(continuation.image, next_step_image)
    chapter = st.session_state.storyteller.continue_story(
        story_plan=st.session_state.story.story_plan,
        story_so_far=st.session_state.story.full_text,
        chapter_nr=n_chapters + 1,
        next_step=next_step_text,
    )
    chapter_translated = st.session_state.translator.translate_to_language(chapter)
    st.session_state.story.add_chapter(
        text=chapter,
        image=next_step_image,
        translated_text=chapter_translated,
    )

    tot_chapters = len(st.session_state.story.story_plan)
    if n_chapters + 1 < tot_chapters:
        add_continuations()
    else:
        st.session_state.story.reset_continuations()
        final_image = st.session_state.illustrator.create_b64_encoded_picture(
            protagonist_description=st.session_state.story.protagonist_description,
            scene=chapter,
        )
        image_data = b64decode(final_image)
        image_path = "images/final.png"
        with open(image_path, "wb") as png:
            png.write(image_data)
        st.session_state.story.final_image = image_path


def display_chapter(chapter: Chapter) -> None:
    cols = st.columns([1, 2])
    with cols[0]:
        if chapter.image:
            st.image(
                chapter.image,
                # width=512,
            )
    with cols[1]:
        st.markdown(chapter.translated_text)


def display_continuations() -> None:
    n_continuations = len(st.session_state.story.possible_continuations)
    cols = st.columns([1] * n_continuations)
    for continuation, col, col_pos in zip(
        st.session_state.story.possible_continuations,
        cols,
        list(range(n_continuations)),
    ):
        with col:
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


def display_final_image() -> None:
    st.image(
        st.session_state.story.final_image,
        "The end",
        width=512,
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
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    language = st.selectbox("Language", ("Italian", "German", "English"), index=None)
    user_input = st.text_input(
        label="What should the story be about?",
        value="Suprise me",
        key="user_story_prompt",
    )
    st.button(
        "Start story",
        key="start_story",
        on_click=start_story,
        args=[language, user_input],
    )

else:
    if st.session_state.story:
        for chapter in st.session_state.story.chapters:
            display_chapter(chapter)

        if st.session_state.story.possible_continuations:
            display_continuations()
        elif st.session_state.story.final_image:
            display_final_image()
