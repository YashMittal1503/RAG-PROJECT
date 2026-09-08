"""
Unit tests for extractive contextual compression in the RAG pipeline.
Verifies that:
1. Distractor sentences are pruned from long chunks while preserving relevant content.
2. Adjacent context windows (±1 sentence) are kept for grammatical continuity.
3. Short chunks (<= 4 sentences) are left untouched.
4. Chunks with no query matches safely fall back to original text.
5. Citation tags and summary chunks are properly preserved.
"""

import pytest
from app.services.query import compress_chunk_content, _build_context


SAMPLE_PROSE_CHUNK = """
Chapter 1: History of Computing.
Early computing devices were mechanical calculators designed in the nineteenth century.
Charles Babbage designed the Analytical Engine in 1837 which used punch cards.
Ada Lovelace wrote the first algorithm intended for execution on the machine.
Decades later, vacuum tubes powered the ENIAC during the Second World War.
In 1971, the Intel 4004 was released as the world's first commercial single-chip microprocessor.
The 4004 had 2,300 transistors and operated at a clock frequency of 740 kilohertz.
It could perform up to 92,600 instructions per second and delivered 4-bit computation.
Later microprocessors like the 8086 introduced the x86 instruction set architecture.
Modern processors pack tens of billions of transistors into a single silicon die.
Cloud computing has shifted workloads to massive distributed data centers worldwide.
""".strip()


def test_compress_chunk_retains_relevant_sentences():
    query = "What were the specifications and performance of the Intel 4004 microprocessor?"
    compressed = compress_chunk_content(SAMPLE_PROSE_CHUNK, query)

    # Must be meaningfully shorter than original
    assert len(compressed) < len(SAMPLE_PROSE_CHUNK)

    # Must contain the relevant facts
    assert "Intel 4004" in compressed
    assert "2,300 transistors" in compressed
    assert "740 kilohertz" in compressed

    # Must exclude unrelated distant sentences
    assert "Charles Babbage" not in compressed
    assert "Ada Lovelace" not in compressed


def test_compress_short_chunk_skipped():
    short_text = "Python is an interpreted language. It is easy to learn. It supports multiple paradigms."
    compressed = compress_chunk_content(short_text, "How does Python work?")
    assert compressed == short_text


def test_compress_no_match_fallback():
    query = "quantum entanglement photon polarization"
    compressed = compress_chunk_content(SAMPLE_PROSE_CHUNK, query)
    # When no query words match, it must safely fall back to full original content
    assert compressed == SAMPLE_PROSE_CHUNK


def test_build_context_applies_compression_and_preserves_citations():
    chunks = [
        {
            "id": "c1",
            "filename": "history.pdf",
            "page_number": 4,
            "chunk_type": "text",
            "content": SAMPLE_PROSE_CHUNK,
        },
        {
            "id": "c2",
            "filename": "summary.csv",
            "chunk_type": "summary",
            "content": "Spreadsheet summary: Total sales by region across 2024.",
        }
    ]

    query = "What was the clock frequency of the 4004 microprocessor?"
    context = _build_context(chunks, query=query, enable_compression=True)

    # Citations tags must be intact
    assert "[Page 4] (Source: history.pdf)" in context
    assert "[Summary] (Source: summary.csv)" in context

    # Text chunk is compressed
    assert "740 kilohertz" in context
    assert "Charles Babbage" not in context

    # Summary chunk is not compressed
    assert "Spreadsheet summary: Total sales by region across 2024." in context


def test_build_context_can_disable_compression():
    chunks = [
        {
            "id": "c1",
            "filename": "history.pdf",
            "page_number": 4,
            "chunk_type": "text",
            "content": SAMPLE_PROSE_CHUNK,
        }
    ]

    query = "What was the clock frequency of the 4004 microprocessor?"
    context = _build_context(chunks, query=query, enable_compression=False)

    # Uncompressed context contains the whole chunk
    assert "Charles Babbage" in context
    assert "740 kilohertz" in context
