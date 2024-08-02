"""
Chat app and UI
"""

from base64 import b64decode

import streamlit as st

from illustrator import Dalle3Illustrator
from story import Chapter, OpenEndedStory
from storyteller import LLMStoryteller

N_ALTERNATIVES = 2


def start_story():
    st.session_state.story = OpenEndedStory()
    st.session_state.storyteller = LLMStoryteller(language=st.session_state.language)
    st.session_state.illustrator = Dalle3Illustrator()
    story_start = st.session_state.storyteller.start_story()
    st.session_state.story.add_chapter(text=story_start)
    # st.session_state.story.append(
    #     {
    #         "text": story_start,
    #         "images": ["images/office.png", "images/office.png", "images/office.png"],
    #         "captions": None,
    #     }
    # )
    # story_so_far = "\n\n".join(chapter["text"] for chapter in st.session_state.story)
    st.session_state.story.protagonist_description = (
        st.session_state.storyteller.describe_protagonist(
            story_so_far=st.session_state.story.full_text
        )
    )
    print("---" * 20)
    print("Protagonist description:")
    print(st.session_state.story.protagonist_description)

    protagonist_image = st.session_state.illustrator.create_picture(
        protagonist_description=st.session_state.story.protagonist_description,
        scene="No scene has been set, just draw the full-body protagonist",
    )
    image_data = b64decode(protagonist_image)
    image_path = "images/protagonist.png"
    with open(image_path, "wb") as png:
        png.write(image_data)
    st.session_state.story.chapters[0].image = image_path

    alternatives = st.session_state.storyteller.propose_alternatives(
        story_so_far=st.session_state.story.full_text, n_alternatives=N_ALTERNATIVES
    )
    # TODO use possible_contnuations and Chapter
    st.session_state.story.alternatives = alternatives

    for j, alternative in enumerate(st.session_state.story.alternatives):
        alternative_image = st.session_state.illustrator.create_picture(
            protagonist_description=st.session_state.story.protagonist_description,
            scene=alternative,
        )
        image_data = b64decode(alternative_image)
        image_path = f"images/alt_{j}.png"
        with open(image_path, "wb") as png:
            png.write(image_data)
        # st.session_state.story.chapters[0].image = image_path

    # for alternative in alternatives:
    #    print(st.session_state.image_describer.describe_picture(scene=alternative))


def write_chapter(next_step):
    # story_so_far = "\n\n".join(chapter["text"] for chapter in st.session_state.story)
    chapter = st.session_state.storyteller.continue_story(
        st.session_state.story.full_text, next_step
    )
    st.session_state.story.add_chapter(text=chapter)
    # st.session_state.story.append(
    #     {
    #         "text": chapter,
    #         "images": ["images/office.png", "images/office.png", "images/office.png"],
    #         "captions": None,
    #     }
    # )
    # story_so_far = "\n\n".join(chapter["text"] for chapter in st.session_state.story)
    alternatives = st.session_state.storyteller.propose_alternatives(
        story_so_far=st.session_state.story.full_text,
        n_alternatives=N_ALTERNATIVES,
    )
    st.session_state.story.alternatives = alternatives

    for j, alternative in enumerate(st.session_state.story.alternatives):
        alternative_image = st.session_state.illustrator.create_picture(
            protagonist_description=st.session_state.story.protagonist_description,
            scene=alternative,
        )
        image_data = b64decode(alternative_image)
        image_path = f"images/alt_{j}.png"
        with open(image_path, "wb") as png:
            png.write(image_data)


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
        st.rerun()
else:
    # st.image(
    #     ["/Users/al.farace/Projects/llm-interactive-story/images/office.png"]
    #     * N_ALTERNATIVES,
    #     caption=st.session_state.story.alternatives,
    #     width=200,
    # )

    if st.session_state.story:
        for chapter in st.session_state.story.chapters:
            if chapter.image:
                st.image(
                    chapter.image,
                    width=256,
                )
            st.markdown(chapter.text)

        if st.session_state.story.alternatives:
            # last_chapter = st.session_state.story[-1]
            cols = st.columns([1] * N_ALTERNATIVES)
            for image, caption, col in zip(
                [f"images/alt_{j}.png" for j in range(N_ALTERNATIVES)],
                st.session_state.story.alternatives,
                cols,
            ):
                with col:
                    st.image(
                        image,
                        caption,
                        width=200,
                    )
                    st.button(
                        "Continue",
                        key=image + str(col),
                        on_click=write_chapter,
                        args=["next step"],
                    )
    # else:
    #    st.button("Start", key="start", on_click=start_story)
