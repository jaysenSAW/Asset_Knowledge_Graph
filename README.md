# Asset_Knowledge_Graph

Analyze political speeches and build a knowledge graph.

## Project structure

```
src/
├── prompts/
│   ├── prompt_RAG_production_V4.txt
│   
├── llm/                                            (reserved, currently empty)
├── graph/
│   └── graph_builder.py
├── preprocessing/
│   └── speeches.py
└── main_build_graphRAG.py

data/
├── discours-presidents/
└── discours_2017-05-14_2026-08-05/

credential.json        # Neo4j credentials, see setup step 0
output/                # created automatically on first run
├── merged_graph/
├── neo4j_graph/
├── log_graph.txt
└── global_report.json
```
---

## 0) Install Neo4j, Ollama and the Qwen model

Before using the pipeline, install the following dependencies.

**Neo4j Desktop**

[https://neo4j.com/download/](https://neo4j.com/download/)

**Ollama** — on Windows, the install command is:

```
irm https://ollama.com/install.ps1 | iex
```

Check that the Ollama server is running:

```
ollama list
```

You can also check the API response directly:

```
curl http://localhost:11434/api/tags
```

If you've just installed Ollama, you won't have any models yet. Pull `qwen2.5:7b` and `qwen2.5-coder:7b`:

```
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

Check that it runs correctly:

```
ollama run qwen2.5:7b
```

You can type anything into the prompt to confirm it responds.

**Neo4j credentials**

Create a `credential.json` file at the project root (the pipeline exits immediately if it's missing):

```json
{
  "URI": "bolt://localhost:7687",
  "USER": "neo4j",
  "PASSWORD": "your-password"
}
```

---

## 1) Knowledge graph schema

Node labels:

```
(:Speech)
(:Chunk)
(:Person)
(:Organization)
(:Institution)
(:Country)
(:City)
(:Event)
(:Theme)
(:Law)
```

Relationship types:

```
:HAS_CHUNK
:LOCATED_IN
:MEMBER_OF
:ALLIED_WITH
:VISITED
:SPOKE_ABOUT
:PROPOSED
:ANNOUNCED
:SUPPORTS
:MENTIONS
```

---

## 2) Pipeline overview

```
                    Discours (TXT)
                          |
                 Parsing / header & footer
                    stripping (speeches.py)
                          |
                      Chunking
              (1200 chars / 200 overlap)
                          |
          +---------------+---------------+
          |                               |
      Embeddings                   LLM extraction
   (batched, background)         (parallel, per chunk)
          |                               |
          |                     Per-chunk validation
          |                  (allowed types/relations,
          |                confidence, evidence actually
          |                 supports source AND target)
          |                               |
          |                        Merge all chunks
          |                               |
          |                  Graph-level validation
          |               (dedup entities/relations,
          |                generate deterministic ids)
          |                               |
          +---------------+---------------+
                          |
                  Build Neo4j graph
                  (Speech / Chunk / Entity
                   nodes + relationships)
                          |
                Save JSON (merged_graph,
                    neo4j_graph)
                          |
                    Load into Neo4j
```

---

## 3) Run the pipeline

### A) Create graphRAG from speeches

**Downloading the corpus**

```
playwright install chromium
```
**Exxtract node and relationship from speeches**

```
python main_build_graphRAG.py -f data/discours-presidents/ -llm qwen2.5:7b -w 4 -s 5
```

Main arguments (see `main_build_graphRAG.py` for the full list and defaults):

| Flag | Description |
|---|---|
| `-f, --folder` | Folder containing the raw speech `.txt` files |
| `-llm, --llm_model` | Ollama model name used for extraction |
| `-r, --prompt_RAG` | Path to the extraction prompt |
| `-n, --neo4j_para` | Path to `credential.json` |
| `-log, --log_path` | Path to the per-speech processing log |
| `-o, --output_folder` | Output folder for generated graph JSON files |
| `-g, --global_report` | Path to the aggregated extraction report |
| `-w, --workers` | Number of parallel workers for LLM chunk extraction |
| `-s, --save_every` | Write the log/report to disk every N speeches |

---

### B) Create graphRAG from neo4j_graph folder


## 4) Downloading the corpus

```
playwright install chromium
```

---

## 5) Limitations

Information is extracted using an LLM. For each relation, the pipeline checks that `confidence == 1.0`, that the `evidence` quote is exact and present in the chunk, and most importantly that the normalized `source` **and** `target` names both actually appear within the `evidence` snippet itself (not just somewhere else in the chunk). 
This last check was added specifically because the LLM sometimes picks a real sentence from the chunk as "evidence" for a relation that sentence doesn't actually support.

Even with these checks, some false relations can pass validation, and some legitimate but less explicit relations get excluded when the wording isn't straightforward. A second LLM acting as a judge could improve accuracy further, at the cost of pipeline speed. For this proof of concept, we chose to keep the pipeline as fast as possible, even if that costs some accuracy.

**Note:** the pipeline currently does **not** resolve nicknames or references (e.g. "le président français", "le chef de l'État") to a canonical `Person` name such as `Emmanuel Macron`. This is a deliberate choice, not a missing feature: the extraction prompt explicitly forbids turning a pronoun or a function/title into a named entity unless that name is written out explicitly in the text, to avoid hallucinating identities. An alias table like the one below is **not** implemented and would go against that rule as currently designed. it's listed here only as a possible future direction if that trade-off is revisited:

```python
aliases = {
    "macron": "Emmanuel Macron",
    "le président français": "Emmanuel Macron",
    "le chef de l'état": "Emmanuel Macron"
}
```