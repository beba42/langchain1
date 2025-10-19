"""Unit tests for local document utilities."""

from langchain_core.documents import Document

from langchain_app.agents.local_reader import chunk_documents


def test_chunk_documents_handles_empty_input():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_documents_returns_documents():
    chunks = chunk_documents(["hello world"], chunk_size=50, chunk_overlap=0)
    assert chunks, "Expected at least one chunk for non-empty input"
    assert isinstance(chunks[0], Document)
