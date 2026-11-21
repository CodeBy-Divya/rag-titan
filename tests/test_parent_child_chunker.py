"""Tests for ParentChildChunker (handbook section 09: parent/child ids, both granularities)."""

import pytest

from rag_titan.chunking import ParentChildChunker, SentenceChunker, WordChunker
from rag_titan.config import get_settings

LEAVE = (
    "# Leave policy\n\n"
    "## Annual leave\n\n"
    "Every full-time employee gets 20 paid leave days per calendar year. "
    "Up to 5 unused days move to the next year.\n\n"
    "## Sick leave\n\n"
    "Tell your manager before 10 am on the first sick day. "
    "From the third sick day you need a note from the doctor.\n"
)
TRAVEL = "# Travel\n\n## Sick leave\n\nSick days abroad follow the same rules.\n"


def children(documents, words=8, overlap=2, parent_size=1500):
    """Chunk documents with small word children."""
    return ParentChildChunker(WordChunker(words, overlap), parent_size).chunk(documents)


def test_every_child_knows_its_parent_section():
    chunks = children([{"source": "leave.md", "text": LEAVE}])
    parents = {chunk["parent_id"]: chunk["parent_text"] for chunk in chunks}

    assert list(parents) == ["leave.md#parent_0", "leave.md#parent_1"]
    assert parents["leave.md#parent_0"].startswith("Leave policy > Annual leave\n\n")
    assert parents["leave.md#parent_1"].startswith("Leave policy > Sick leave\n\n")
    assert len(chunks) > len(parents)          # several small children per parent


def test_children_stay_inside_their_parent():
    for chunk in children([{"source": "leave.md", "text": LEAVE}]):
        assert chunk["text"] in chunk["parent_text"]
        assert chunk["section_path"][-1] in chunk["parent_text"]


def test_child_offsets_point_into_the_document():
    for chunk in children([{"source": "leave.md", "text": LEAVE}]):
        piece = LEAVE[chunk["start_offset"]:chunk["end_offset"]]
        assert piece.split() == chunk["text"].split()


def test_parent_offsets_point_at_the_section_in_the_document():
    for chunk in children([{"source": "leave.md", "text": LEAVE}]):
        section = LEAVE[chunk["parent_start_offset"]:chunk["parent_end_offset"]]
        assert chunk["parent_text"].endswith(section.strip())
        assert chunk["parent_start_offset"] <= chunk["start_offset"]
        assert chunk["end_offset"] <= chunk["parent_end_offset"]


def test_pdf_parents_know_their_pages():
    text = "Leave rules start here.\n\nMore leave rules.\fPage two goes on."
    chunks = children([{"source": "hr.pdf", "text": text}], words=4, overlap=0)
    assert chunks[0]["parent_page_numbers"] == [1, 2]
    assert chunks[0]["page_numbers"] == [1]


def test_ids_are_unique_across_files_with_the_same_section_title():
    chunks = children([{"source": "leave.md", "text": LEAVE},
                       {"source": "travel.md", "text": TRAVEL}])

    assert [chunk["id"] for chunk in chunks] == [f"chunk_{n}" for n in range(len(chunks))]
    travel = {chunk["parent_id"] for chunk in chunks if chunk["source"] == "travel.md"}
    assert travel == {"travel.md#parent_0"}
    assert "Travel > Sick leave" in chunks[-1]["parent_text"]


def test_a_long_section_becomes_several_parents():
    text = "# Rules\n\n" + "\n\n".join(f"Rule {n} says something useful here." for n in range(12))
    chunks = children([{"source": "rules.md", "text": text}], parent_size=120)

    parents = {chunk["parent_id"]: chunk["parent_text"] for chunk in chunks}
    assert len(parents) > 1
    assert all(len(parent) <= 120 for parent in parents.values())


def test_text_without_headings_gets_paragraph_parents():
    text = "First paragraph about leave.\n\nSecond paragraph about travel."
    chunks = children([{"source": "notes.txt", "text": text}], parent_size=40)

    assert {chunk["parent_id"] for chunk in chunks} == {"notes.txt#parent_0", "notes.txt#parent_1"}
    assert "section_path" not in chunks[0]


def test_any_chunker_can_make_the_children():
    chunker = ParentChildChunker(SentenceChunker(80, 0))
    chunks = chunker.chunk([{"source": "leave.md", "text": LEAVE}])
    assert chunks[0]["text"].startswith("Every full-time employee")
    assert chunker.split_text(LEAVE) == [chunk["text"] for chunk in chunks]


def test_empty_document_gives_no_chunks():
    assert children([{"source": "empty.md", "text": "  \n\n "}]) == []


@pytest.mark.parametrize("size", [0, -5, 1.5, "big", True])
def test_bad_parent_size_raises(size):
    with pytest.raises(ValueError, match="parent_size"):
        ParentChildChunker(WordChunker(8, 2), parent_size=size)


def test_child_chunker_must_be_a_plain_chunker():
    with pytest.raises(ValueError, match="child_chunker"):
        ParentChildChunker("word")
    with pytest.raises(ValueError, match="another ParentChildChunker"):
        ParentChildChunker(ParentChildChunker(WordChunker(8, 2)))


def test_parent_size_setting(monkeypatch):
    assert get_settings()["parent_size"] == 1500
    monkeypatch.setenv("PARENT_SIZE", "800")
    assert get_settings()["parent_size"] == 800
    monkeypatch.setenv("PARENT_SIZE", "big")
    with pytest.raises(ValueError, match="PARENT_SIZE"):
        get_settings()
