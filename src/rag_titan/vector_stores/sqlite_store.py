"""
Vector store saved in one SQLite file: index once, ask again tomorrow.

Uses only Python's built-in sqlite3 and numpy, so nothing extra is installed.
Search compares the question with every stored vector (exact cosine), which is
fast enough for up to roughly 100,000 chunks.

Example:
    from rag_titan.vector_stores import SQLiteStore

    store = SQLiteStore("rag_titan.db")
"""

import json
from pathlib import Path

import numpy as np

from rag_titan.vector_stores.base_store import BaseStore

TABLES = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    chunk TEXT NOT NULL,          -- the chunk dict as JSON
    vector BLOB NOT NULL          -- float32 numbers
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks (source);
CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class SQLiteStore(BaseStore):
    """
    Keeps chunks and vectors in a SQLite file that survives restarts.

    Args:
        path: the database file, created (with its folder) when missing.
    """

    def __init__(self, path="rag_titan.db"):
        # imported here: the browser (Pyodide) has no sqlite3 unless it is loaded
        import sqlite3

        self.path = Path(path)
        if self.path.is_dir():
            raise ValueError(f"SQLiteStore needs a file path, but {self.path} is a folder")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.connection = sqlite3.connect(str(self.path))
            self.connection.executescript(TABLES)
        except sqlite3.DatabaseError as error:
            raise ValueError(f"{self.path} is not a rag-titan SQLite index: {error}") from None
        self._cache = None          # (data_version, chunks, matrix), see _load()

    def __repr__(self):
        return f"SQLiteStore({str(self.path)!r})"

    def add(self, chunks, vectors):
        """Save each chunk with its vector (a chunk with the same id is replaced)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        rows = [(chunk["id"], chunk["source"], json.dumps(chunk, ensure_ascii=False),
                 np.asarray(vector, dtype=np.float32).tobytes())
                for chunk, vector in zip(chunks, vectors)]
        self._check_sizes(rows)

        with self.connection:       # one transaction: all rows or none
            self.connection.executemany(
                "INSERT OR REPLACE INTO chunks (id, source, chunk, vector) VALUES (?, ?, ?, ?)",
                rows)

    def search(self, query_vector, top_k=5):
        """Compare the query with every chunk and return the best top_k."""
        chunks, matrix = self._load()
        if not chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        dots = matrix @ query
        # a zero vector has no direction: score it 0 instead of dividing by zero
        scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)

        best = np.argsort(-scores, kind="stable")[:top_k]
        return [{**chunks[i], "score": float(scores[i])} for i in best]

    def clear(self):
        """Remove every stored chunk and the saved info."""
        with self.connection:
            self.connection.execute("DELETE FROM chunks")
            self.connection.execute("DELETE FROM info")

    def count(self):
        """Return how many chunks are stored."""
        return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def delete_source(self, source):
        """Remove every chunk of one file and return how many were removed."""
        with self.connection:
            cursor = self.connection.execute("DELETE FROM chunks WHERE source = ?", (source,))
        return cursor.rowcount

    def get_info(self):
        """Return how the index was built, or None for an empty store."""
        row = self.connection.execute("SELECT value FROM info WHERE key = 'index'").fetchone()
        return json.loads(row[0]) if row else None

    def set_info(self, info):
        """Remember how the index was built."""
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO info (key, value) VALUES "
                                    "('index', ?)", (json.dumps(info),))

    def close(self):
        """Close the database file (it also closes when Python exits)."""
        self.connection.close()

    def _load(self):
        """
        Return (chunks, matrix of vectors), read from the file only when it changed.

        SQLite's data_version goes up when another connection (another program,
        or another SQLiteStore on the same file) writes, so the copy in memory
        is never out of date.
        """
        version = self.connection.execute("PRAGMA data_version").fetchone()[0]
        total_changes = self.connection.total_changes      # writes by this connection
        if self._cache and self._cache[0] == (version, total_changes):
            return self._cache[1], self._cache[2]

        rows = self.connection.execute("SELECT chunk, vector FROM chunks ORDER BY rowid").fetchall()
        chunks = [json.loads(chunk) for chunk, _ in rows]
        matrix = (np.stack([np.frombuffer(vector, dtype=np.float32) for _, vector in rows])
                  if rows else np.zeros((0, 0), dtype=np.float32))
        self._cache = ((version, total_changes), chunks, matrix)
        return chunks, matrix

    def _check_sizes(self, rows):
        """All vectors in one index must have the same length."""
        sizes = {len(row[3]) for row in rows}
        stored = self.connection.execute("SELECT vector FROM chunks LIMIT 1").fetchone()
        if stored:
            sizes.add(len(stored[0]))
        if len(sizes) > 1:
            dims = sorted(size // 4 for size in sizes)     # 4 bytes per float32
            raise ValueError(f"Every vector in one index must have the same length, got {dims}")
