"""Local file reading agent stub."""

from pathlib import Path
from typing import Sequence
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_app.config.settings import settings


def load_local_documents(pattern: str = "*.txt", encodings: Sequence[str] = ("utf-8", "utf-8-sig", "latin-1")) -> list[str]:
    """Load text documents from the data directory.

    Attempts multiple encodings so we can ingest files saved with legacy code pages.
    """
    data_path = settings.data_dir
    documents: list[str] = []

    for file_path in data_path.glob(pattern):
        if not file_path.is_file():
            continue

        text = None
        for encoding in encodings:
            try:
                text = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            warnings.warn(f"Skipping {file_path} due to unsupported encoding.", RuntimeWarning)
            continue

        documents.append(text)

    return documents


def chunk_documents(documents: list[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Chunk documents for embedding or retrieval workflows."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.page_content for chunk in splitter.create_documents(documents)]
