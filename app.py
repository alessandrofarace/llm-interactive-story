"""
Chat app and UI
"""

import logging
from base64 import b64decode

import streamlit as st

from illustrator import MockIllustrator
from story import OpenEndedStory
from storyteller import LLMStoryteller


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

N_ALTERNATIVES = 2


def start_story():
    st.session_state.story = OpenEndedStory()
    st.session_state.storyteller = LLMStoryteller(language=st.session_state.language)
    st.session_state.illustrator = MockIllustrator()
    story_start = st.session_state.storyteller.start_story()
    st.session_state.story.add_chapter(text=story_start)
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
        story_so_far=st.session_state.story.full_text, n_alternatives=N_ALTERNATIVES
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
        st.session_state.story.add_continuation(
            text=continuation_text,
            image=image_path,
        )
    st.rerun()


def write_chapter(next_step):
    chapter = st.session_state.storyteller.continue_story(
        st.session_state.story.full_text, next_step
    )
    st.session_state.story.add_chapter(text=chapter)
    possible_continuations = st.session_state.storyteller.propose_continuation(
        story_so_far=st.session_state.story.full_text, n_alternatives=N_ALTERNATIVES
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
        st.session_state.story.add_continuation(
            text=continuation_text,
            image=image_path,
        )


if "story" not in st.session_state:
    st.session_state.story = None
if "language" not in st.session_state:
    st.session_state.language = None
if "storyteller" not in st.session_state:
    st.session_state.storyteller = None
if "illustrator" not in st.session_state:
    st.session_state.illustrator = None


if st.session_state.language is None:
    language = st.selectbox("Language", ("Italian", "German", "English"), index=None)
    if language is not None:
        st.session_state.language = language
        start_story()
else:
    if st.session_state.story:
        for chapter in st.session_state.story.chapters:
            if chapter.image:
                st.image(
                    chapter.image,
                    width=512,
                )
            st.markdown(chapter.text)

        if st.session_state.story.possible_continuations:
            n_possibilities = len(st.session_state.story.possible_continuations)
            cols = st.columns([1] * n_possibilities)
            for continuation, col in zip(
                st.session_state.story.possible_continuations,
                cols,
            ):
                with col:
                    st.image(
                        continuation.image,
                        continuation.text,
                        width=200,
                    )
                    st.button(
                        "Continue",
                        key=f"write_chapter_{str(col)}",
                        on_click=write_chapter,
                        args=[continuation.text],
                    )
