from enum import Enum
from io import BytesIO
from typing import List

from chainlit.types import AskFileResponse
from pydantic import BaseModel
from pypdf import PdfReader


class FileType(Enum):
    TXT = "text/plain"
    PDF = "application/pdf"


class FileLoader(BaseModel):
    documents: List[AskFileResponse]
    encoding: str = "utf-8"

    def load(self):
        for document in self.documents:
            if document.type == FileType.TXT.value:
                yield self._load_txt(document)
            elif document.type == FileType.PDF.value:
                yield self._load_pdf(document)

    def _load_txt(self, document):
        return document.content.decode(self.encoding)

    def _load_pdf(self, document):
        reader = PdfReader(BytesIO(document.content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


class CharacterTextSplitter(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def _split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_generator(self, text_generator) -> List[str]:
        chunks = []
        for text in text_generator:
            chunks.extend(self._split(text))
        return chunks
