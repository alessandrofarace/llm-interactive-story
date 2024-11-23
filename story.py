from pathlib import Path


class Chapter:

    def __init__(
        self,
        text: str,
        image: Path | None = None,
        translated_text: str | None = None,
    ) -> None:
        self.text = text
        self.image = image
        self.translated_text = translated_text


class Story:

    def __init__(
        self,
        abstract: str | None = None,
        story_plan: dict | None = None,
        chapters: list[Chapter] | None = None,
        protagonist_description: str | None = None,
        final_image: Path | None = None,
    ) -> None:
        self.abstract = abstract
        if story_plan is None:
            story_plan = {}
        self.story_plan = story_plan
        if chapters is None:
            chapters = []
        self.chapters = chapters
        self.protagonist_description = protagonist_description
        self.final_image = final_image

    def add_chapter(
        self,
        text: str = None,
        image: Path = None,
        translated_text: str | None = None,
    ) -> None:
        self.chapters.append(
            Chapter(text=text, image=image, translated_text=translated_text)
        )

    @property
    def full_text(self) -> str:
        return "\n\n".join([chapter.text for chapter in self.chapters])


class OpenEndedStory(Story):

    def __init__(
        self,
        abstract: str | None = None,
        story_plan: dict | None = None,
        chapters: list[Chapter] | None = None,
        protagonist_description: str | None = None,
        possible_continuations: list[Chapter] | None = None,
        final_image: Path | None = None,
    ) -> None:
        super().__init__(
            abstract=abstract,
            story_plan=story_plan,
            chapters=chapters,
            protagonist_description=protagonist_description,
            final_image=final_image,
        )
        if possible_continuations is None:
            possible_continuations = []
        self.possible_continuations = possible_continuations

    def add_continuation(
        self,
        text: str = None,
        image: Path = None,
        translated_text: str | None = None,
    ) -> None:
        self.possible_continuations.append(
            Chapter(
                text=text,
                image=image,
                translated_text=translated_text,
            )
        )

    def reset_continuations(self) -> None:
        self.possible_continuations = []
