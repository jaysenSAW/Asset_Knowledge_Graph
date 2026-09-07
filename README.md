# Asset_Knowledge_Graph

Analyze political speeches and build a knowledge graph.

# Project structure

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

# SCRIPTS

**main_query_graphRAG.py**

Translate questions into Cypher queries, search the database, and generate natural language answers.

**build_graph_from_JSON.py** 

Inject extracted graph JSON files into an empty Neo4j database.

**pipeline_build_graphRAG.py:**

Processes raw speech text files, chunks them, computes dense vector embeddings, extracts entities and relations using an LLM in parallel, merges and validates the resulting subgraphs, and loads the output directly into Neo4j.


# Prerequisites

Before using the pipeline, install the following dependencies.

**Neo4j Desktop**

[https://neo4j.com/download/](https://neo4j.com/download/)

**Ollama**

on Windows, the install command is:

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

If you've just installed Ollama, you won't have any models yet. Pull `qwen2.5:7b`, `qwen2.5-coder:7b` and `mistral:7b-instruct`:

```
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull mistral:7b-instruct
```

Check that it runs correctly:

```
ollama run qwen2.5:7b
```

You can type anything into the prompt to confirm it responds.

**Neo4j credentials**

Start **Neo4j Desktop** create your *username* and *paswword* then create a `credential.json` file at the project root (the pipeline exits immediately if it's missing) and fill it with your username and password:

```json
{
  "URI": "bolt://localhost:7687",
  "USER": "neo4j",
  "PASSWORD": "your-password"
}
```

---

# End-to-End Quickstart Workflow
```
# 1. Build and populate knowledge graph from pre-generated graph JSON files
python build_graph_from_JSON.py -c credential.json -g output/neo4j_graph

# 2. Run CLI Query
python main_query_graphRAG.py --question "Montre-moi les 2 discours les mieux notés qui parlent de la transition écologique" --debug_mode True

# 3. Launch Web Interface
streamlit run app.py
```

# 1) Knowledge graph schema

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

# 2) Pipeline overview

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

# 3) Query the GraphRAG Pipeline

Ensure Ollama is running locally with required models (qwen2.5-coder:7b and qwen2.5:7b).

Translates natural language questions into Cypher queries via LLM, queries Neo4j (using vector and graph traversal), and synthesizes answers via LLM.

`python main_query_graphRAG.py --question "Comment le président compte améliorer le quotidien des français"`


* --folder: Folder containing input speech files. (default: "data/discours-presidents/")
* --llm_cypher: Model used for text-to-Cypher translation. (default: "qwen2.5-coder:7b")
* --llm_chat: Model used for final answer change it to 'mistral:7b-instruct'. (default: "qwen2.5:7b")
* --question: Question to ask the GraphRAG pipeline.
* --schema_prompt_path: Prompt file defining Cypher schema & rules. (Default: "src/prompts/NEO4J_SCHEMA_PROMPT.txt")
* --neo4j_para: Path to Neo4j credentials file. (Default: "credential.json")
* --chatbot_mode: Synthesize answers in natural language (True/False). (Default: True)
* --debug_mode: Display raw Cypher query before execution (True/False).. (Default: False)

## Semantic search

Because the sentence is converted into a vector during the embedding step, the semantic search results depend on the wording of the question. Indeed, the representation of words in the vector space depends on the corpus used during the model's training.

It is therefore better to avoid acronyms and ambiguous terms. For example, **JO** could refer to **Jeux Olympiques** in French or **Justice Office** in English. Using more explicit terms can help reduce ambiguity and improve the search results.

Another important parameter is the similarity threshold. The threshold can be adjusted to retrieve more or fewer results. In our case, the semantic search is based on cosine similarity. Sentences with similar meanings should have a high cosine similarity score, with a maximum value of 1.

You can define the minimum confidence score (cosine similarity) required to retrieve a chunk. For example, when we require a score greater than or equal to 0.8, no relevant result is returned:

* "Les préparatifs pour les Jeux Olympique 2024 à Paris ? Base toi sur un indice de confiance supérieur ou égal à 0.8"

```
Réponse finale : 
Emmanuel Macron
```

By using a less restrictive threshold, we can retrieve relevant results. For example:

* "python .\main_query_graphRAG.py --question "Les préparatifs pour les Jeux Olympique 2024 à Paris ? Base toi sur un indice de confiance supérieur ou égal à 0.7" --debug_mode True"

```
Réponse finale :
Il semble que ces extraits proviennent d'une série de déclarations du président Emmanuel Macron sur les Jeux Olympiques et Paralympiques de Paris 2024. Voici un résumé des points clés extrait de ces déclarations :

1. **Engagement Collectif**:
   - Le président souligne l'importance de l'engagement collectif pour la réussite des Jeux. Il mentionne la participation de l'État, des collectivités territoriales (ville de Paris, région Île-de-France, métropole, département de Seine-Saint-Denis, etc.), et même la Polynésie française.

2. **Préparation et Travail**:
   - Il rappelle que la préparation des Jeux a commencé dès 2017, bien avant la candidature officielle. Le président a passé du temps à préparer la cérémonie d'ouverture et à examiner les dossiers avec les ministres et les préfets.

3. **Héritage et Impact Durable**:
   - Le président insiste sur l'importance de laisser un héritage durable. Il mentionne des projets comme la construction de 5 ponts pour recoudre le lien entre la capitale et la Seine-Saint-Denis, et l'amélioration de l'accessibilité des gares et des commerces.

4. **Tourisme et Événements**:
   - Il souligne que 2024 sera un millésime français avec des événements importants comme le débarquement en Normandie et en Provence, ainsi que les Jeux Olympiques et Paralympiques.

5. **Sport et Développement Durable**:
   - Le président souligne l'importance du sport dans le développement durable et l'universalisme. Il mentionne le Sommet du CIO et l'importance de construire des ponts universels à travers les organisations internationales.

6. **Fierté et Confiance**:
   - Il exprime une grande fierté pour avoir obtenu la candidature des Jeux en 2017 et remercie les élus et les services de l'État pour leur confiance.

Ces déclarations montrent l'engagement du président Emmanuel Macron à assurer une organisation réussie des Jeux Olympiques et Paralympiques de Paris 2024, en mettant l'accent sur l'importance de l'engagement collectif et de la durabilité.
```

This illustrates the importance of choosing an appropriate similarity threshold. A threshold that is too high may result in no retrieved chunks, while a lower threshold can increase recall but may also introduce less relevant results.

## Troubleshot

Sometime with no reason qwen does not reply in french but in chinese or even in english. 

For example the command line return:
Chinese answer:

* `python .\main_query_graphRAG.py --question "la guerre en Ukraine et son impact sur le quotidien" --debug_mode True`

English answer:

* `python .\main_query_graphRAG.py --question "le conflit en Ukraine et son impact sur le quotidien" --debug_mode True`

You can change the final LLM call with the option `--llm_chat`. Use `'mistral:7b-instruct'` fix the problem 

# 4) Build Neo4j Graph

## A) Build Neo4j Graph from Preprocessed JSON Files

Ensure Ollama is running locally with required models (qwen2.5-coder:7b and qwen2.5:7b).

Use build_graph_from_JSON.py to loads pre-generated graph JSON files into an empty Neo4j database without re-running LLM extraction steps.

`python build_graph_from_JSON.py -c credential.json -g output/neo4j_graph`

Main arguments (see `main_build_graphRAG.py` for the full list and defaults):

* --neo4j_para: Path to Neo4j connection configuration JSON file (default 'credential.json').
* --neo4j_JSON: Directory containing JSON graph files to load into Neo4j (default 'output/neo4j_graph').
* --neo4j_constraint: Cypher script file defining Neo4j indexes and uniqueness constraints (default 'src/preprocessing/graph_constraint.txt').


## B) Create graphRAG from speeches

### Downloading the corpus

```
playwright install chromium
```
### Extract nodes and relationship from speeches


Run the script from your terminal:

```
python pipeline_build_graphRAG.py -f data/discours-presidents/ -w 4 --llm_model qwen2.5:7b
```

Main arguments (see `pipeline_build_graphRAG.py` for the full list and defaults):

* --folder: Folder containing input speech files. (default: "data/discours-presidents/")
* --llm_model: Ollama LLM model used for entity and relation extraction. (default: "qwen2.5:7b")
* --prompt_RAG: Prompt template for information extraction. (Default: "src/prompts/prompt_RAG_production_v4.txt")
* --neo4j_para: Path to Neo4j credentials file. (Default: "credential.json")
* --log_path: File path to write processing status logs. (Default: "output/log_graph.txt")
* --output_folder: Base output directory for processed subgraphs. (Default: "output")
* --global_report: Path to write aggregated quality report. (Default: "output/global_report.json")
* --workers: Number of parallel threads for chunk extraction. (Default: 4)
* --save_every: Checkpoint frequency: write logs/reports every N speeches. (Default: 1)
* --beging: Start index for slicing corpus list. (Default: 0)
* --end: End index for slicing corpus list (-1 for all). (Default: -1)
---

**Limit:**

Information is extracted using an LLM. For each relation, the pipeline checks that `confidence == 1.0`, that the `evidence` quote is exact and present in the chunk, and most importantly that the normalized `source` **and** `target` names both actually appear within the `evidence` snippet itself (not just somewhere else in the chunk). 
This last check was added specifically because the LLM sometimes picks a real sentence from the chunk as "evidence" for a relation that sentence doesn't actually support.

Even with these checks, some false relations can pass validation, and some legitimate but less explicit relations get excluded when the wording isn't straightforward. A second LLM acting as a judge could improve accuracy further, at the cost of pipeline speed. For this proof of concept, we chose to keep the pipeline as fast as possible, even if that costs some accuracy.

**Note:** the pipeline currently does **not** resolve nicknames or references (e.g. "le président français", "le chef de l'État") to a canonical `Person` name such as `Emmanuel Macron`. This is a deliberate choice, not a missing feature: the extraction prompt explicitly forbids turning a pronoun or a function/title into a named entity unless that name is written out explicitly in the text, to avoid hallucinating identities. An alias table like the one below is **not** implemented and would go against that rule as currently designed. it's listed here only as a possible future direction if that trade-off is revisited:

# 5) Interactive web application

Run the Streamlit web interface to execute GraphRAG queries, view natural language answers, and visually inspect source subgraphs in real time.

`streamlit run app.py`