"""
Chat app and UI
"""

import streamlit as st

from storyteller import LLMStoryteller


def write_chapter(next_step):
    story_so_far = "\n\n".join(chapter["text"] for chapter in st.session_state.story)
    chapter = st.session_state.storyteller.continue_story(story_so_far, next_step)
    st.session_state.story.append(
        {
            "text": chapter,
            "images": ["images/office.png", "images/office.png", "images/office.png"],
            "captions": None,
        }
    )
    story_so_far = "\n\n".join(chapter["text"] for chapter in st.session_state.story)
    alternatives = st.session_state.storyteller.propose_alternatives(
        story_so_far=story_so_far
    ).split("\n")
    alternatives = [el for el in alternatives if len(el) > 0]
    print(alternatives)
    st.session_state.story[-1]["captions"] = alternatives


def start_story():
    story_start = st.session_state.storyteller.start_story()
    st.session_state.story.append(
        {
            "text": story_start,
            "images": ["images/office.png", "images/office.png", "images/office.png"],
            "captions": None,
        }
    )
    story_so_far = "\n\n".join(chapter["text"] for chapter in st.session_state.story)
    alternatives = st.session_state.storyteller.propose_alternatives(
        story_so_far=story_so_far
    ).split("\n")
    print(alternatives)
    alternatives = [el for el in alternatives if len(el) > 0]
    st.session_state.story[-1]["captions"] = alternatives


if "story" not in st.session_state:
    st.session_state.story = []
if "language" not in st.session_state:
    st.session_state.language = None
if "storyteller" not in st.session_state:
    st.session_state.storyteller = None

if st.session_state.language is None:
    language = st.selectbox("Language", ("Italian", "German", "English"), index=None)
    st.session_state.language = language
    st.session_state.storyteller = LLMStoryteller(language=language)
    start_story()
    if language is not None:
        st.rerun()
else:

    # st.title("Storyteller")

    for chapter in st.session_state.story:
        st.markdown(chapter["text"])
        st.image(
            chapter["images"],
            caption=chapter["captions"],
            width=200,
        )

    if st.session_state.story:
        last_chapter = st.session_state.story[-1]
        cols = st.columns([1, 1, 1, 1])
        for image, caption, col in zip(
            last_chapter["images"], last_chapter["captions"], cols
        ):
            with col:
                st.button(
                    "Continue",
                    key=image + str(col),
                    on_click=write_chapter,
                    args=[caption],
                )
    # else:
    #    st.button("Start", key="start", on_click=start_story)
