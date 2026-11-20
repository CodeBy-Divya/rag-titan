# Configuration

Settings are read from **environment variables** or a **`.env` file** in the
folder where you run your code. Every setting has a default, so nothing is required.

![Settings pick a class for every part](assets/images/architecture.svg)

Copy the example file to start:

```bash
cp .env.example .env
```

## All settings

| Variable          | Settings key      | Default                  | Meaning                                        |
|-------------------|-------------------|--------------------------|------------------------------------------------|
| `LLM_PROVIDER`    | `llm_provider`    | `ollama`                 | `ollama`, or `claude` / `anthropic` (below)    |
| `LLM_MODEL`       | `llm_model`       | provider's (`gemma3:4b`) | Chat model for answers                         |
| `OLLAMA_HOST`     | `ollama_host`     | `http://localhost:11434` | Ollama server address                          |
| `ANTHROPIC_API_KEY` | –               | –                        | Your Claude key (`claude` only)                |
| `AGENT_MODEL`     | `agent_model`     | provider's (`qwen3:4b`)  | Model for agents (needs tools)                 |
| `AGENT_THINK`     | `agent_think`     | `true`                   | Turn on thinking for agents                    |
| `AGENT_MAX_STEPS` | `agent_max_steps` | `5`                      | Max agent think/tool rounds                    |
| `EMBED_PROVIDER`  | `embed_provider`  | `ollama`                 | `ollama`, or `hash` (no model, see below)      |
| `EMBED_MODEL`     | `embed_model`     | `nomic-embed-text`       | Embedding model                                |
| `CHUNKER`         | `chunker`         | `word`                   | see [Chunking methods](chunking.md)            |
| `CHUNK_SIZE`      | `chunk_size`      | `150`                    | Units per chunk (see below)                    |
| `CHUNK_OVERLAP`   | `chunk_overlap`   | `30`                     | Units shared by neighbouring chunks            |
| `CHUNK_ENCODING`  | `chunk_encoding`  | `cl100k_base`            | tiktoken encoding (`token` only)               |
| `VECTOR_STORE`    | `vector_store`    | `memory`                 | Where vectors are stored                       |
| `DEDUPE_OVERLAP`  | `dedupe_overlap`  | `0.5`                    | Drop near-copy search results; `off` keeps all |
| `RERANKER`        | `reranker`        | `keyword`                | `keyword` or `none`                            |
| `GENERATOR`       | `generator`       | `grounded`               | How answers are written                        |
| `LOG_LEVEL`       | –                 | `INFO`                   | Used by `setup_logging()`                      |

`CHUNKER` is one of `word`, `character`, `token`, `recursive`, `sentence`,
`paragraph`, `heading` or `page`. `CHUNK_SIZE` and `CHUNK_OVERLAP` count words
for `word`, model tokens for `token`, and characters for all the others. Change them together
when you switch: 150 words is roughly 200 tokens or 900 characters, so for
`recursive`, `paragraph` and `heading` try `CHUNK_SIZE=1000`, for `sentence`
`500`, and for `page` `2000`. `CHUNKER=token` needs
`pip install "rag-titan[tokens]"`; to use your own tokenizer, see
[Chunking methods](chunking.md#your-own-tokenizer).

## Ollama Cloud

The same settings work with [Ollama Cloud](https://ollama.com/cloud) models.
Either keep using your local Ollama (run `ollama signin` once, then pick a
`-cloud` model), or talk to the cloud directly with an API key:

```bash
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=your-key          # sent for you as "Authorization: Bearer your-key"
LLM_MODEL=gpt-oss:20b-cloud
```

Cloud models need usage credits on your Ollama account; without them a request
fails with "Add usage credits or subscribe" (status 402). Keep `EMBED_MODEL`
on a local model if your plan has no cloud embeddings.

## Claude (Anthropic)

Let Claude write the answers instead of a local model. Nothing extra to install.

1. Get an API key at [console.anthropic.com](https://console.anthropic.com).
2. Put it in your `.env` (never commit this file) and pick the provider:

    ```bash
    LLM_PROVIDER=claude              # "anthropic" works too
    ANTHROPIC_API_KEY=sk-ant-...
    # LLM_MODEL=claude-sonnet-5-5    # optional, default: claude-haiku-5-5
    ```

3. Use rag-titan as before, or choose in code:

    ```python
    from rag_titan import RAGTitan

    rag = RAGTitan(settings={"llm_provider": "claude"})
    rag = RAGTitan(settings={"llm_provider": "claude", "llm_model": "claude-sonnet-5-5"})
    ```

| Provider | Answers (`LLM_MODEL`)  | Agents (`AGENT_MODEL`) |
|----------|------------------------|------------------------|
| `ollama` | `gemma3:4b`            | `qwen3:4b`             |
| `claude` | `claude-haiku-5-5`     | `claude-haiku-5-5`     |

These defaults are used only when you leave `LLM_MODEL` / `AGENT_MODEL` unset,
so changing `LLM_PROVIDER` alone is enough.

Good to know:

- **Search still uses the embedder.** Claude writes the answers; turning text into
  vectors stays with `EMBED_PROVIDER` (Ollama's `nomic-embed-text` by default, or
  `hash` to run without Ollama at all).
- **Your chunks leave your machine.** The question and the chunks found for it are
  sent to Anthropic. Keep `ollama` for documents that must stay local.
- **Agents work too:** Claude supports tool calling. `AGENT_THINK` only applies to Ollama.
- **Errors are plain:** a missing key says to set `ANTHROPIC_API_KEY`; a wrong key or
  model shows Anthropic's own message (the key itself is never printed).
- In Python, `AnthropicLLM(model, api_key=None, max_tokens=2048, temperature=None)`
  from `rag_titan.llm` can be used directly (see [Building blocks](building-blocks.md)).

## Without an embedding model

`EMBED_PROVIDER=hash` uses `HashEmbedder`: every word is hashed into a vector,
so no model is needed. Search then matches **words**, not meaning ("holiday"
does not find "leave"). It is meant for trying rag-titan offline, for tests and
for the browser playground, not for real documents.

## Changing settings in code

Pass a dict. Your values win over the environment and the defaults.

```python
from rag_titan import RAGTitan, get_settings

rag = RAGTitan(settings={"llm_model": "llama3.2", "chunk_size": 200})

settings = get_settings({"reranker": "none"})   # full settings dict
```

## Logging

`rag-titan` prints nothing until you ask it to:

```python
from rag_titan import setup_logging

setup_logging("DEBUG")   # DEBUG, INFO or WARNING
```
