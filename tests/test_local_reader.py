"""Unit tests for local document utilities."""

from langchain_app.agents.local_reader import chunk_documents


def test_chunk_documents_handles_empty_input():
    chunks = chunk_documents([])
    assert chunks == []

