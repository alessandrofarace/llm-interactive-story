from pathlib import Path


class Chapter:

    def __init__(self, text: str, image: Path | None = None) -> None:
        self.text = text
        self.image = image


class Story:

    def __init__(
        self,
        chapters: list[Chapter] | None = None,
        protagonist_description: str | None = None,
    ) -> None:
        if chapters is None:
            chapters = []
        self.chapters = chapters
        self.protagonist_description = protagonist_description

    def add_chapter(self, text: str = None, image: Path = None) -> None:
        self.chapters.append(Chapter(text=text, image=image))

    @property
    def full_text(self) -> str:
        return "\n\n".join([chapter.text for chapter in self.chapters])


class OpenEndedStory(Story):

    def __init__(
        self,
        chapters: list[Chapter] | None = None,
        protagonist_description: str | None = None,
        possible_continuations: list[Chapter] | None = None,
    ) -> None:
        super().__init__(
            chapters=chapters, protagonist_description=protagonist_description
        )
        if possible_continuations is None:
            possible_continuations = []
        self.possible_continuations = possible_continuations
