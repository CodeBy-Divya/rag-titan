"""
Fit the search step of a RAG type's FLOW to the retriever it uses.

A RAG type draws one "retrieve" node. A retriever can show itself there:
    - with a `label` (DenseRetriever: "Vector search"), the node is renamed;
    - with a SEARCH_FLOW of its own steps (HybridRetriever: vector search ->
      keyword search -> fuse), the node is replaced by those steps.
So every RAG type gets the right diagram for every retriever, without code of its own.
"""

SEARCH_NODE = "retrieve"


def expand_search(flow, retriever):
    """
    Return a copy of `flow` with its search node fitted to `retriever`.

    Args:
        flow: a FLOW dict ({"nodes": [...], "edges": [...]}).
        retriever: the retriever the RAG type uses (None leaves the flow as it is).
    """
    nodes = [dict(node) for node in flow["nodes"]]
    edges = [list(edge) for edge in flow["edges"]]
    if retriever is None or SEARCH_NODE not in {node["id"] for node in nodes}:
        return {"nodes": nodes, "edges": edges}

    steps = getattr(retriever, "SEARCH_FLOW", None)
    if steps is None:
        label = getattr(retriever, "label", None)
        for node in nodes:
            if node["id"] == SEARCH_NODE and label:
                node["label"] = label
        return {"nodes": nodes, "edges": edges}

    # the retriever's steps take the node's place: arrows in go to its first
    # step, arrows out leave from its last step
    first, last = steps["nodes"][0]["id"], steps["nodes"][-1]["id"]
    expanded = []
    for node in nodes:
        expanded.extend([dict(step) for step in steps["nodes"]] if node["id"] == SEARCH_NODE
                        else [node])
    rewired = [[last if a == SEARCH_NODE else a, first if b == SEARCH_NODE else b]
               for a, b in edges]
    return {"nodes": expanded, "edges": rewired + [list(edge) for edge in steps["edges"]]}
