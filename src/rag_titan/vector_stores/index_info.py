"""
Remember how an index was built, and stop searches that would mix things up.

Vectors from different embedding models cannot be compared: a question
embedded with "hash" finds nonsense in an index built with "nomic-embed-text".
A store that saves this info (see BaseStore.get_info) lets rag-titan say so
clearly instead of returning wrong chunks.
"""


def build_info(embedder, chunker, dimensions, next_chunk):
    """
    Describe how an index is built.

    Args:
        embedder: the BaseEmbedder used for the vectors.
        chunker: the BaseChunker used for the chunks.
        dimensions: length of every vector.
        next_chunk: the number the next chunk id will get ("chunk_<n>").

    Returns:
        A dict of plain values that every store can save.
    """
    return {
        "embed_model": str(embedder.model),
        "dimensions": int(dimensions),
        "chunker": type(chunker).__name__,
        "next_chunk": int(next_chunk),
    }


def check_same_build(info, embed_model, dimensions):
    """
    Raise a clear error when the index was built with another embedder.

    Args:
        info: what store.get_info() returned (None skips the check).
        embed_model: name of the embedding model in use now.
        dimensions: length of the vectors it makes.
    """
    if not info:
        return

    old_model, old_dimensions = info.get("embed_model"), info.get("dimensions")
    if old_model == str(embed_model) and old_dimensions == int(dimensions):
        return

    raise ValueError(
        f"This index was built with {old_model} ({old_dimensions} dims), but you "
        f"are using {embed_model} ({int(dimensions)} dims). Index again with "
        "reset=True, or switch EMBED_PROVIDER / EMBED_MODEL back."
    )
