"""
`rag-titan index` and the saved-index options of `ask` and `chat`.

Index once, then ask as often as you like without the files:
    rag-titan index --files handbook.pdf --vector-store sqlite
    rag-titan index --files new_policy.pdf --vector-store sqlite --add
    rag-titan ask --question "How many leave days?" --vector-store sqlite
"""

from rag_titan.app import RAGTitan
from rag_titan.vector_stores.store_registry import VECTOR_STORES


def add_store_options(command):
    """Add --vector-store and --store-path, which win over VECTOR_STORE(_PATH) in .env."""
    command.add_argument("--vector-store", choices=VECTOR_STORES,
                         help="memory (forgotten after the command) or sqlite (saved); "
                              "default: VECTOR_STORE")
    command.add_argument("--store-path", help="the saved index file (sqlite); "
                                              "default: VECTOR_STORE_PATH or rag_titan.db")


def add_index_command(commands):
    """Describe `rag-titan index`."""
    index = commands.add_parser("index", help="index files once into a saved vector store")
    index.add_argument("--files", nargs="+", required=True, help="PDF, DOCX, MD or TXT files")
    index.add_argument("--add", action="store_true",
                       help="add to the saved index (a file indexed before is replaced) "
                            "instead of starting over")
    index.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")
    add_store_options(index)


def store_settings(args):
    """The settings the store options change."""
    settings = {}
    if args.vector_store:
        settings["vector_store"] = args.vector_store
    if args.store_path:
        settings["vector_store_path"] = args.store_path
    return settings


def run_index(args):
    """Index the files into the saved store and say where they went."""
    rag = RAGTitan(settings=store_settings(args))
    store = rag.pipeline.store
    if not store.persistent:
        raise SystemExit("rag-titan index: VECTOR_STORE=memory keeps nothing after the "
                         "command. Use --vector-store sqlite (or set VECTOR_STORE=sqlite).")

    added = rag.index(args.files, reset=not args.add)
    print(f"Indexed {added} chunks from {len(args.files)} file(s) into {where(rag)}; "
          f"it now holds {store.count()} chunks.")


def prepare_rag(args):
    """
    Build the RAGTitan for `ask` and `chat`.

    With --files the files are indexed first (into a saved store this
    replaces what it held). Without --files the saved index is used.
    """
    rag = RAGTitan(rag_type=args.rag_type, settings=store_settings(args))
    store = rag.pipeline.store

    if args.files:
        chunk_count = rag.index(args.files)
        print(f"Indexed {chunk_count} chunks from {len(args.files)} file(s).")
        return rag

    if not store.persistent:
        raise SystemExit(f"rag-titan {args.command}: nothing is saved with "
                         "VECTOR_STORE=memory. Pass --files, or index once with "
                         "`rag-titan index --files ... --vector-store sqlite`.")
    if store.count() == 0:
        raise SystemExit(f"rag-titan {args.command}: nothing is indexed in {where(rag)} yet. "
                         "Run `rag-titan index --files ...` first.")

    print(f"Using {store.count()} chunks saved in {where(rag)}.")
    return rag


def where(rag):
    """Name the saved index for messages, for example "rag_titan.db"."""
    return str(getattr(rag.pipeline.store, "path", rag.settings["vector_store"]))
