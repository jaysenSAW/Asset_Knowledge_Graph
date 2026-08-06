import hashlib
import json
from neo4j import GraphDatabase
import ollama
import os
from sentence_transformers import SentenceTransformer
import unicodedata


def create_entity_id(entity_type, name):
    """create unique id by using couple
    entity_type and name

    Args:
        entity_type (string): _description_
        name (string): _description_

    Returns:
        hashlib.sha256(): entity's id
    """
    value = f"{entity_type}:{name.lower()}"

    return hashlib.sha256(
        value.encode()
    ).hexdigest()

def normalize_name(name):
    """Normalize string by removing special character

    Args:
        name (string): 

    Returns:
        name: normalized string
    """

    name = name.lower().strip()
    name = unicodedata.normalize(
        "NFKD",
        name
    )
    return name

def get_node_id(entity: dict) -> str:
    """
    Create deterministic id for each entity

    Parameters
    ----------
    entity : dict
        Exemple :
        {
            "type": "Person",
            "name": "Emmanuel Macron"
        }

    Returns
    -------
    str
        Exemple :
        Person_4e5b2c8f7d6a...
    """

    entity_type = entity["type"]
    entity_name = normalize_name(entity["name"])

    key = f"{entity_type}:{entity_name}"

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

    return f"{entity_type}_{digest}"


def extract_graph(prompt,
                  model = "qwen2.5:7b"):

    # prompt = prompt_builder(prompt_template, example_json, text)

    response = ollama.chat(
        model = model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )
    content = response["message"]["content"]
    try:
        graph = json.loads(content)
    except json.JSONDecodeError:
        print("Erreur JSON")
        print(content)
        return None
    return graph

def compute_chunk_embeddings(
        chunks, 
        embedding_model = SentenceTransformer("BAAI/bge-m3")) -> list:

    for chunk in chunks:
        chunk["embedding"] = embedding_model.encode(
            chunk["text"],
            normalize_embeddings=True
        ).tolist()

    return chunks

def merge_graphs(graph1, graph2):
    """
    Fusionne deux Knowledge Graphs.

    Les entités sont fusionnées par (type, name).
    Les relations sont fusionnées par
    (source, relation, target).

    Parameters
    ----------
    graph1 : dict
    graph2 : dict

    Returns
    -------
    dict
    """

    merged = {
        "entities": [],
        "relations": []
    }

    ##################################################
    # ENTITIES
    ##################################################

    entity_index = {}

    for entity in graph1["entities"] + graph2["entities"]:

        key = (
            entity["type"],
            entity["name"].strip().lower()
        )

        if key not in entity_index:
            entity_index[key] = entity

    merged["entities"] = list(entity_index.values())

    ##################################################
    # RELATIONS
    ##################################################

    relation_index = {}

    for relation in graph1["relations"] + graph2["relations"]:

        key = (
            relation["source"].strip().lower(),
            relation["relation"],
            relation["target"].strip().lower()
        )

        if key not in relation_index:

            relation_index[key] = relation

        else:

            # garder la meilleure confidence
            if relation["confidence"] > relation_index[key]["confidence"]:
                relation_index[key] = relation

    merged["relations"] = list(relation_index.values())

    return merged


# graph curration

def remove_duplicate_entities(entities : list[dict], report : dict):
    """
    """
    entity_index = {}
    for entity in entities:
        key = (
            entity["type"],
            normalize_name(entity["name"])
        )
        if key not in entity_index:
            entity["id"] = get_node_id(entity)
            entity_index[key] = entity
        else:
            #duplicate entity
            report["duplicate_entities"] += 1

    entities = list(entity_index.values())
    return entities, report

def check_relationships(entity_by_name : dict, relations : list[dict], report : dict) -> dict:
    relation_index = {}

    for relation in relations:

        confidence = relation.get("confidence", 0)

        if confidence < 0.8:
            report["low_confidence_relations"] += 1
            continue

        source = normalize_name(relation["source"])
        target = normalize_name(relation["target"])

        if source not in entity_by_name:
            report["missing_source"] += 1
            report["missing_sources"].append(
                relation["source"]
            )
            continue

        if target not in entity_by_name:
            report["missing_target"] += 1
            report["missing_targets"].append(
                relation["target"]
            )
            continue

        relation["source_id"] = entity_by_name[source]["id"]
        relation["target_id"] = entity_by_name[target]["id"]

        key = (
            relation["source_id"],
            relation["relation"],
            relation["target_id"]
        )

        if key not in relation_index:

            relation_index[key] = relation

        else:
            report["duplicate_relations"] += 1
            # keep the relation with highest confidence
            if relation["confidence"] > relation_index[key]["confidence"]:
                relation_index[key] = relation
    return relation_index, report


def validate_graph(graph):
    """
    Validate and clean a graph extracted by the LLM.

    Operations
    ----------
    - remove duplicated entities
    - remove duplicated relations
    - remove relations with confidence < 0.8
    - remove relations referencing missing entities
    - generate deterministic node ids
    - generate source_id / target_id

    Parameters
    ----------
    graph : dict

    Returns
    -------
    dict
    """
    report = {
        "initial_entities": len(graph["entities"]),
        "initial_relations": len(graph["relations"]),

        "final_entities": 0,
        "final_relations": 0,

        "duplicate_entities": 0,
        "duplicate_relations": 0,

        "low_confidence_relations": 0,
        "missing_source": 0,
        "missing_target": 0,

        "removed_relations": 0,

        "missing_sources": [],
        "missing_targets": []
    }

    ##################################################
    # 1. Remove duplicated entities
    ##################################################

    entities, report = remove_duplicate_entities(graph["entities"], report)

    ##################################################
    # 2. Index by normalized name
    ##################################################

    entity_by_name = {
        normalize_name(e["name"]): e
        for e in entities
    }

    ##################################################
    # 3. Validate relations
    ##################################################

    relation_index, report = check_relationships(entity_by_name, graph["relations"], report)

    ##################################################
    # 4. Build final graph and report
    ##################################################

    validated_graph = {
        "entities": entities,
        "relations": list(relation_index.values())
    }
    report["final_entities"] = len(
        validated_graph["entities"]
    )

    report["final_relations"] = len(
        validated_graph["relations"]
    )

    report["removed_relations"] = (
        report["initial_relations"]
        - report["final_relations"]
    )

    report["missing_sources"] = sorted(
        set(report["missing_sources"])
    )

    report["missing_targets"] = sorted(
        set(report["missing_targets"])
    )

    return validated_graph, report

def feed_global_report(report, speech, global_report = None):
    if global_report is None:
        global_report = {
        "speeches": 0,
        "chunks": 0,
        "entities": 0,
        "relations": 0,
        "duplicate_entities": 0,
        "duplicate_relations": 0,
        "missing_targets": {},
        "missing_sources": {}
        }
    global_report["speeches"] += 1
    global_report["chunks"] += len(speech["chunks"])
    global_report["entities"] += report["final_entities"]
    global_report["relations"] += report["final_relations"]
    global_report["duplicate_entities"] += report["duplicate_entities"]
    global_report["duplicate_relations"] += report["duplicate_relations"]

    for target in report["missing_targets"]:
        global_report["missing_targets"][target] = (
            global_report["missing_targets"].get(target, 0) + 1
        )

    for source in report["missing_sources"]:
        global_report["missing_sources"][source] = (
            global_report["missing_sources"].get(source, 0) + 1
        )
    return global_report


def build_neo4j_graph(speech, merged_graph):
    """
    Convert a validated graph into a Neo4j graph structure.

    Parameters
    ----------
    speech : dict
        Speech metadata + chunks

    merged_graph : dict
        Output of validate_graph()

    Returns
    -------
    graph : dict
    """
    graph = {
        "nodes": [],
        "relationships": []
    }
    # Speech node
    speech_node = {
        "id": speech["id"],
        "label": "Speech",
        "properties": {
            "title": speech["title"],
            "date": speech["date"],
            "filename": speech["filename"],
            "header": speech["header"]
        }
    }
    graph["nodes"].append(speech_node)
    # Chunk nodes
    for chunk in speech["chunks"]:
        chunk_id = hashlib.sha256(
            f"{speech['id']}_{chunk['chunk_id']}".encode()
        ).hexdigest()
        graph["nodes"].append({
            "id": f"Chunk_{chunk_id}",
            "label": "Chunk",
            "properties": {
                "speech_id": speech["id"],
                "chunk_number": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": chunk["embedding"]
            }
        })
        graph["relationships"].append({
            "type": "HAS_CHUNK",
            "source": speech["id"],
            "target": f"Chunk_{chunk_id}",
            "properties": {}
        })
    # Entity nodes
    for entity in merged_graph["entities"]:
        graph["nodes"].append({
            "id": entity["id"],
            "label": entity["type"],
            "properties": {
                "name": entity["name"]
            }
        })
    # Graph relations
    for relation in merged_graph["relations"]:
        graph["relationships"].append({
            "type": relation["relation"],
            "source": relation["source_id"],
            "target": relation["target_id"],
            "properties": {
                "confidence": relation["confidence"],
                "evidence": relation["evidence"]
            }
        })
    # Chunk -> Entity (MENTIONS)
    for chunk in speech["chunks"]:
        chunk_id = hashlib.sha256(
            f"{speech['id']}_{chunk['chunk_id']}".encode()
        ).hexdigest()
        chunk_text = chunk["text"].lower()
        for entity in merged_graph["entities"]:
            if entity["name"].lower() in chunk_text:
                graph["relationships"].append({
                    "type": "MENTIONS",
                    "source": f"Chunk_{chunk_id}",
                    "target": entity["id"],
                    "properties": {}
                })
    return graph

def save_graph(graph, output_file):
    """export database as json

    Args:
        graph (dict): database to save
        output_file (str): file name
    """
    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            graph,
            f,
            ensure_ascii=False,
            indent=2
        )


def load_to_neo4j(driver, graph):
    """
    Insert graph into Neo4j.

    Parameters
    ----------
    driver : neo4j.Driver

    graph : dict
    """

    with driver.session() as session:

        ##########################################
        # Nodes
        ##########################################

        for node in graph["nodes"]:

            cypher = f"""
            MERGE (n:{node["label"]} {{id:$id}})
            SET n += $properties
            """

            session.run(
                cypher,
                id=node["id"],
                properties=node["properties"]
            )

        ##########################################
        # Relationships
        ##########################################

        for rel in graph["relationships"]:

            cypher = f"""
            MATCH (a {{id:$source}})
            MATCH (b {{id:$target}})

            MERGE (a)-[r:{rel["type"]}]->(b)

            SET r += $properties
            """
            session.run(
                cypher,
                source=rel["source"],
                target=rel["target"],
                properties=rel["properties"]
            )


def create_constraints(driver, constraint_file : str):
    with driver.session() as session:
        with open(constraint_file, "r", encoding="utf-8") as f:
            queries = f.read()
        for query in queries.split(";"):
            query = query.strip()
            if query:
                session.run(query)
    print("Neo4j constraints created.")