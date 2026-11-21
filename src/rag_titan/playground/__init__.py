"""
The RAG playground on the docs site: the real package runs in the browser (Pyodide)
and replays model runs recorded with tools/playground/record_traces.py.
"""

from rag_titan.playground.live import LivePlayground
from rag_titan.playground.presets import (
    PRESET_CHUNKING,
    PRESET_KEYS,
    build_parts,
    index_documents,
    preset_settings,
    replay_parts,
    with_retriever,
)
from rag_titan.playground.recording import (
    PRESET_GRID,
    check_presets,
    grid_tunings,
    record_presets,
    summarize_run,
)
from rag_titan.playground.session import Playground

__all__ = [
    "LivePlayground",
    "PRESET_CHUNKING",
    "PRESET_GRID",
    "PRESET_KEYS",
    "Playground",
    "build_parts",
    "check_presets",
    "grid_tunings",
    "index_documents",
    "preset_settings",
    "record_presets",
    "replay_parts",
    "summarize_run",
    "with_retriever",
]
