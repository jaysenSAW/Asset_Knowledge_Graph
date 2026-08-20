from collections import defaultdict
import hashlib
import json
from neo4j import GraphDatabase
import ollama
import os
import re
from sentence_transformers import SentenceTransformer
import unicodedata

# Optional dependency (pip install json-repair --break-system-packages).
# json-repair is used as a fallback when the LLM's JSON output got cut off 
# mid-string by the num_predict cap: 
# repair_json can usually close the unterminated string/object and salvage 
# the entities/relations that were already generated before the truncation point, 
# instead of discarding the whole chunk. 
try:
    from json_repair import repair_json
    _JSON_REPAIR_AVAILABLE = True
except ImportError:
    _JSON_REPAIR_AVAILABLE = False


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
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
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


def extract_graph(prompt, text, model="qwen2.5:7b", debug=False):

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0,
            # A valid graph for a ~1200-char chunk should never need more
            # than a few hundred tokens. Without a cap, the model can
            # occasionally fall into a repetition loop and keep generating
            # (tens of thousands of tokens of near-duplicate
            # entities/relations) until it's cut off mid-object, producing
            # a multi-hundred-KB response that fails json.loads() anyway
            # and costs several minutes per chunk for nothing. Capping
            # num_predict makes that failure mode fail in seconds instead.
            # 3000 (raised from 2000) gives some headroom for chunks with
            # many entities/relations and long evidence quotes, while
            # staying ~10x below the size of the runaway generations this
            # cap was introduced to prevent (~140KB responses).
            "num_predict": 3000
        },
        keep_alive=-1
    )

    content = response["message"]["content"]

    if debug:
        print("=" * 100)
        print("RÉPONSE BRUTE OLLAMA")
        print(content)
        print("=" * 100)

    try:
        graph = json.loads(content)
    except json.JSONDecodeError as e:
        print("Erreur JSON :", e)

        # Fallback: the output may just be truncated (e.g. cut off
        # mid-string by num_predict) rather than genuinely malformed.
        # Try to repair and salvage whatever entities/relations were
        # already generated before giving up on the chunk entirely.
        if not _JSON_REPAIR_AVAILABLE:
            return None

        try:
            repaired = repair_json(content, return_objects=True)
        except Exception:
            return None

        if not (
            isinstance(repaired, dict)
            and "entities" in repaired
            and "relations" in repaired
        ):
            return None

        print("JSON tronqué réparé avec succès")
        graph = repaired

    # Validation du JSON produit par le LLM
    validated_graph, validation_report = validate_llm_output(
        graph,
        text
    )

    return validated_graph


# Cache so the (heavy) embedding model is loaded only once per process,
# no matter how many times compute_chunk_embeddings is called, and without
# relying on a mutable/expensive default argument evaluated at import time.
_embedding_model_cache = {}

def extract_graph_debug(prompt, model="qwen2.5:7b"):

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        },
        keep_alive=-1
    )

    content = response["message"]["content"]

    # print("\n RAW LLM OUTPUT")
    # print(content)
    # print("====================================")

    try:
        graph = json.loads(content)
    except json.JSONDecodeError as e:
        print("Erreur JSON :", e)
        return None

    print("Graph type:", type(graph))
    print("Graph keys:", graph.keys() if isinstance(graph, dict) else None)

    if isinstance(graph, dict):
        print("Entities:", len(graph.get("entities", [])))
        print("Relations:", len(graph.get("relations", [])))

    return graph

def get_embedding_model(model_name="BAAI/bge-m3"):
    if model_name not in _embedding_model_cache:
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
    return _embedding_model_cache[model_name]


def compute_chunk_embeddings(
        chunks,
        embedding_model=None,
        batch_size=32) -> list:
    """Compute embeddings for a list of chunks in a single batched call
    instead of one call per chunk. Much faster, especially on GPU, since
    the model processes several texts together instead of paying the
    per-call overhead for each chunk individually.
    """
    if not chunks:
        return chunks

    if embedding_model is None:
        embedding_model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks

def merge_graphs(graphs):

    merged = {
        "entities": [],
        "relations": []
    }

    entity_keys = set()
    relation_keys = set()

    # print(f"\nMerging {len(graphs)} graphs")

    for i, graph in enumerate(graphs):

        if graph is None:
            print(f"Graph {i}: None")
            continue

        # print(
        #     f"Graph {i}: "
        #     f"{len(graph.get('entities', []))} entities, "
        #     f"{len(graph.get('relations', []))} relations"
        # )

        for entity in graph.get("entities", []):

            key = (
                entity["type"],
                normalize_name(entity["name"])
            )

            if key not in entity_keys:
                entity_keys.add(key)
                merged["entities"].append(entity)

        for relation in graph.get("relations", []):

            key = (
                normalize_name(relation["source"]),
                relation["relation"],
                normalize_name(relation["target"])
            )

            if key not in relation_keys:
                relation_keys.add(key)
                merged["relations"].append(relation)

    # print(
    #     f"MERGED: "
    #     f"{len(merged['entities'])} entities, "
    #     f"{len(merged['relations'])} relations"
    # )

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
            if relation.get("confidence", 0) > relation_index[key].get("confidence", 0):
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


def add_speaker_entities(merged_graph, speech):
    """Add a Person entity for each of the speech's speaker(s) (see
    speeches.extract_speakers), so a discourse's speaker is present in the
    graph even when the LLM never gets an explicit name to attach a
    SPOKE_ABOUT-style relation to (e.g. "je" is never resolved to a name
    by design — see validate_llm_output's forbidden_person_names).

    This does NOT go through validate_llm_output: it's not an LLM guess,
    it's a deterministic read of metadata the source document explicitly
    labels. It uses the same get_node_id() scheme as LLM-extracted
    entities, so if the same person is also explicitly named somewhere in
    the chunk text and extracted separately, both end up as the same
    Neo4j node (MERGE on id) instead of duplicates.

    Parameters
    ----------
    merged_graph : dict
        Output of validate_graph() — entities/relations already validated

    speech : dict
        Speech metadata, expected to have a "speakers" list (possibly
        empty) as produced by speeches.load_speeches

    Returns
    -------
    dict
        merged_graph, with speaker Person entities appended if not
        already present
    """
    existing_ids = {
        entity["id"] for entity in merged_graph["entities"]
        if "id" in entity
    }

    for name in speech.get("speakers", []):
        entity = {"type": "Person", "name": name}
        entity["id"] = get_node_id(entity)

        if entity["id"] not in existing_ids:
            merged_graph["entities"].append(entity)
            existing_ids.add(entity["id"])

    return merged_graph


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

    # Speech -> speaker(s): deterministic, from document metadata (see
    # add_speaker_entities), not an LLM guess. The Person node itself is
    # expected to already be in merged_graph["entities"] (added via
    # add_speaker_entities before this function is called) and will be
    # picked up by the entity loop below like any other entity — this
    # just adds the relationship linking it to the speech.
    for name in speech.get("speakers", []):
        speaker_id = get_node_id({"type": "Person", "name": name})
        graph["relationships"].append({
            "type": "DELIVERED_BY",
            "source": speech["id"],
            "target": speaker_id,
            "properties": {}
        })

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
                "confidence": relation.get("confidence"),
                "evidence": relation.get("evidence", "")
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


_LABEL_RE = re.compile(r"[^A-Za-z0-9_]")


def sanitize_label(label: str) -> str:
    """Turn an arbitrary string (label or relationship type coming from
    the LLM output) into something safe to interpolate into a Cypher
    query. Neo4j doesn't support parameterized labels/relationship types,
    so this string interpolation can't be avoided, but it must never
    contain anything the LLM could use to inject extra Cypher.
    """
    safe = _LABEL_RE.sub("_", label or "Unknown")
    if not safe or safe[0].isdigit():
        safe = f"L_{safe}"
    return safe


def load_to_neo4j(driver, graph):
    """
    Insert graph into Neo4j.

    Nodes and relationships are grouped by label / (source label, type,
    target label) and written with UNWIND, so each group is a single
    round trip instead of one round trip per node/relationship. Matching
    relationship endpoints by label (instead of just {id: ...}) also lets
    Neo4j use the per-label uniqueness constraint/index on `id` instead of
    scanning every node in the graph.

    Parameters
    ----------
    driver : neo4j.Driver

    graph : dict
    """

    node_label_by_id = {
        node["id"]: node["label"] for node in graph["nodes"]
    }

    with driver.session() as session:

        ##########################################
        # Nodes, grouped by label
        ##########################################

        nodes_by_label = defaultdict(list)
        for node in graph["nodes"]:
            nodes_by_label[node["label"]].append({
                "id": node["id"],
                "properties": node["properties"]
            })

        for label, nodes in nodes_by_label.items():
            safe_label = sanitize_label(label)
            cypher = f"""
            UNWIND $nodes AS node
            MERGE (n:{safe_label} {{id: node.id}})
            SET n += node.properties
            """
            session.run(cypher, nodes=nodes)

        ##########################################
        # Relationships, grouped by
        # (source label, type, target label)
        ##########################################

        rels_by_key = defaultdict(list)
        for rel in graph["relationships"]:
            source_label = sanitize_label(
                node_label_by_id.get(rel["source"], "")
            )
            target_label = sanitize_label(
                node_label_by_id.get(rel["target"], "")
            )
            rel_type = sanitize_label(rel["type"])

            rels_by_key[(source_label, rel_type, target_label)].append({
                "source": rel["source"],
                "target": rel["target"],
                "properties": rel["properties"]
            })

        for (source_label, rel_type, target_label), rels in rels_by_key.items():
            cypher = f"""
            UNWIND $rels AS rel
            MATCH (a:{source_label} {{id: rel.source}})
            MATCH (b:{target_label} {{id: rel.target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += rel.properties
            """
            session.run(cypher, rels=rels)


def create_constraints(driver, constraint_file : str):
    with driver.session() as session:
        with open(constraint_file, "r", encoding="utf-8") as f:
            queries = f.read()
        for query in queries.split(";"):
            query = query.strip()
            if query:
                session.run(query)
    print("Neo4j constraints created.")

ALLOWED_ENTITY_TYPES = {
    "Person",
    "Organization",
    "Institution",
    "Country",
    "City",
    "Law",
    "Event",
    "Theme",
}

ALLOWED_RELATIONS = {
    "MET_WITH",
    "VISITED",
    "LOCATED_IN",
    "MEMBER_OF",
    "PART_OF",
    "SPOKE_ABOUT",
    "PROPOSED",
    "ANNOUNCED",
    "SIGNED",
    "SUPPORTS",
    "OPPOSES",
    "MENTIONS",
}

def validate_llm_output(graph: dict, chunk_text: str) -> tuple[dict, dict]:
    """
    Validate the raw graph returned by the LLM against the original chunk.

    This validation is deliberately performed BEFORE merge_graphs() because
    the original chunk text is required to detect hallucinated entities,
    invalid relations and invalid evidence.

    Returns
    -------
    cleaned_graph : dict
        Graph containing only valid entities and relations.

    report : dict
        Validation statistics.
    """

    report = {
        "initial_entities": 0,
        "initial_relations": 0,

        "valid_entities": 0,
        "valid_relations": 0,

        "removed_entities": 0,
        "removed_relations": 0,

        "hallucinated_entities": 0,
        "forbidden_entities": 0,
        "invalid_entity_types": 0,

        "invalid_relations": 0,
        "missing_relation_source": 0,
        "missing_relation_target": 0,
        "invalid_evidence": 0,
        "evidence_does_not_support_relation": 0,
        "invalid_relation_type": 0,

        "removed_entity_names": [],
        "invalid_evidence_values": []
    }

    if not isinstance(graph, dict):
        return {
            "entities": [],
            "relations": []
        }, report

    chunk_text = chunk_text or ""

    # Normalized once here instead of inside the entity loop below: it was
    # being recomputed for every single entity in the chunk, which is
    # wasted work proportional to n_entities * len(chunk_text).
    normalized_chunk_text = normalize_name(chunk_text)

    # ---------------------------------------------------------
    # Allowed values (module-level constants, not redefined here)
    # ---------------------------------------------------------

    allowed_entity_types = ALLOWED_ENTITY_TYPES
    allowed_relations = ALLOWED_RELATIONS

    # normalize_name() applies NFKD decomposition (accented characters
    # become base letter + separate combining mark), so a plain literal
    # like "les français" written in source (precomposed form) will NOT
    # string-equal normalize_name("les Français") even though they look
    # identical — the comparison silently fails for every accented entry.
    # Running each literal through normalize_name() here keeps both sides
    # in the same decomposed form.
    forbidden_person_names = {
        normalize_name(n) for n in [
            "je",
            "j'",
            "j",
            "moi",
            "nous",
            "vous",
            "il",
            "elle",
            "ils",
            "elles",
            "on",
            "les français",
            "les françaises et les français",
            "le président",
            "la présidente",
            "le ministre",
            "la ministre",
            "le gouvernement",
            "notre président",
            "nos compatriotes",
            "le peuple français"
        ]
    }

    # ---------------------------------------------------------
    # 1. Validate entities
    # ---------------------------------------------------------

    valid_entities = []
    entity_names = set()

    for entity in graph.get("entities", []):

        report["initial_entities"] += 1

        if not isinstance(entity, dict):
            report["removed_entities"] += 1
            continue

        entity_type = entity.get("type")
        entity_name = entity.get("name")

        if not isinstance(entity_type, str):
            report["removed_entities"] += 1
            report["invalid_entity_types"] += 1
            continue

        if not isinstance(entity_name, str):
            report["removed_entities"] += 1
            continue

        entity_type = entity_type.strip()
        entity_name = entity_name.strip()

        if not entity_name:
            report["removed_entities"] += 1
            continue

        # -----------------------------------------------------
        # Invalid entity type
        # -----------------------------------------------------

        if entity_type not in allowed_entity_types:
            report["removed_entities"] += 1
            report["invalid_entity_types"] += 1
            report["removed_entity_names"].append(entity_name)
            continue

        normalized_name = normalize_name(entity_name)

        # -----------------------------------------------------
        # Forbidden Person
        # -----------------------------------------------------

        if (
            entity_type == "Person"
            and normalized_name in forbidden_person_names
        ):
            report["removed_entities"] += 1
            report["forbidden_entities"] += 1
            report["removed_entity_names"].append(entity_name)
            continue

        # -----------------------------------------------------
        # Entity must actually occur in the chunk
        # -----------------------------------------------------

        if normalized_name not in normalized_chunk_text:
            report["removed_entities"] += 1
            report["hallucinated_entities"] += 1
            report["removed_entity_names"].append(entity_name)
            continue

        # -----------------------------------------------------
        # Deduplicate inside the LLM response
        # -----------------------------------------------------

        key = (
            entity_type,
            normalized_name
        )

        if key in entity_names:
            report["removed_entities"] += 1
            continue

        entity_names.add(key)

        # Keep ONLY the fields expected from the LLM
        valid_entities.append({
            "type": entity_type,
            "name": entity_name
        })

    report["valid_entities"] = len(valid_entities)

    # ---------------------------------------------------------
    # Build name index
    #
    # IMPORTANT:
    # relations use names, not type + name.
    # ---------------------------------------------------------

    entity_by_name = {
        normalize_name(entity["name"]): entity
        for entity in valid_entities
    }

    # ---------------------------------------------------------
    # 2. Validate relations
    # ---------------------------------------------------------

    valid_relations = []
    relation_keys = set()

    for relation in graph.get("relations", []):

        report["initial_relations"] += 1

        if not isinstance(relation, dict):
            report["removed_relations"] += 1
            report["invalid_relations"] += 1
            continue

        source = relation.get("source")
        relation_type = relation.get("relation")
        target = relation.get("target")
        evidence = relation.get("evidence")
        confidence = relation.get("confidence")

        # -----------------------------------------------------
        # Required fields
        # -----------------------------------------------------

        if not all([
            isinstance(source, str),
            isinstance(relation_type, str),
            isinstance(target, str),
            isinstance(evidence, str)
        ]):
            report["removed_relations"] += 1
            report["invalid_relations"] += 1
            continue

        source = source.strip()
        relation_type = relation_type.strip()
        target = target.strip()
        evidence = evidence.strip()

        # -----------------------------------------------------
        # Relation type
        # -----------------------------------------------------

        if relation_type not in allowed_relations:
            report["removed_relations"] += 1
            report["invalid_relation_type"] += 1
            continue

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        if confidence != 1.0:
            report["removed_relations"] += 1
            report["invalid_relations"] += 1
            continue

        # -----------------------------------------------------
        # Source must exist in validated entities
        # -----------------------------------------------------

        source_key = normalize_name(source)
        target_key = normalize_name(target)

        if source_key not in entity_by_name:
            report["removed_relations"] += 1
            report["missing_relation_source"] += 1
            continue

        # -----------------------------------------------------
        # Target must exist in validated entities
        # -----------------------------------------------------

        if target_key not in entity_by_name:
            report["removed_relations"] += 1
            report["missing_relation_target"] += 1
            continue

        # -----------------------------------------------------
        # Evidence must exist EXACTLY in chunk
        # -----------------------------------------------------

        if evidence not in chunk_text:
            report["removed_relations"] += 1
            report["invalid_evidence"] += 1
            report["invalid_evidence_values"].append(evidence)
            continue

        # -----------------------------------------------------
        # Evidence must actually support THIS relation: both the
        # source and target names must appear within the evidence
        # snippet itself, not just somewhere else in the chunk.
        # Without this, the LLM can (and does) pick a real sentence
        # from the chunk as "evidence" for a relation that sentence
        # never actually states — e.g. evidence="l'audace de la
        # liberté, l'exigence de l'égalité, la volonté de la
        # fraternité" (no mention of "France") used to justify
        # France -MENTIONS-> "l'audace de la liberté". The prompt
        # asks the model not to do this, but a 7B model at
        # temperature 0 doesn't reliably self-enforce it, so it's
        # enforced here instead.
        # -----------------------------------------------------

        normalized_evidence = normalize_name(evidence)

        if (
            source_key not in normalized_evidence
            or target_key not in normalized_evidence
        ):
            report["removed_relations"] += 1
            report["evidence_does_not_support_relation"] += 1
            report["invalid_evidence_values"].append(evidence)
            continue

        # -----------------------------------------------------
        # Prevent duplicate relation
        # -----------------------------------------------------

        relation_key = (
            source_key,
            relation_type,
            target_key
        )

        if relation_key in relation_keys:
            report["removed_relations"] += 1
            continue

        relation_keys.add(relation_key)

        valid_relations.append({
            "source": source,
            "relation": relation_type,
            "target": target,
            "evidence": evidence,
            "confidence": 1.0
        })

    report["valid_relations"] = len(valid_relations)

    cleaned_graph = {
        "entities": valid_entities,
        "relations": valid_relations
    }

    return cleaned_graph, report