import ollama
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import re

def text2vector(text : str, model : SentenceTransformer = None) -> list:
    """convert text to vector

    Args:
        text (str): text
        model (SentenceTransformer, optional): encoder use to transform texxt to vector. Defaults to SentenceTransformer("BAAI/bge-m3").

    Returns:
        list[np.ndarray]: embedded vector
    """
    if model is None:
        model = SentenceTransformer("BAAI/bge-m3")
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    # Convert into list
    return embedding.tolist()

def create_vector_index(driver):
    cypher_query = """
    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
      }
    }
    """
    with driver.session() as session:
        session.run(cypher_query)
    print("Index vectoriel 'chunk_embeddings' check/create.")

def get_schema_from_driver(driver):
    """_summary_

    Args:
        driver (_type_): _description_

    Returns:
        _type_: _description_
    """
    with driver.session() as session:
        result = session.run("CALL db.schema.visualization()")
        record = result.single()
        nodes = [f"(:{node['name']})" for node in record["nodes"]]
        relationships = [
            f"(:{rel.start_node['name']})-[:{rel.type}]->(:{rel.end_node['name']})"
            for rel in record["relationships"]
        ]
        return f"Nodes:\n" + "\n".join(nodes) + "\n\nRelationships:\n" + "\n".join(relationships)



def generate_cypher_query_app(user_question: str, NEO4J_SCHEMA_PROMPT: str, model: str = "qwen2.5-coder:7b") -> str:
    """_summary_

    Args:
        user_question (str): _description_
        model (_type_, optional): _description_. Defaults to "qwen2.5-coder:7b".

    Returns:
        str: _description_
    """
    
    messages = [
        {"role": "system", "content": NEO4J_SCHEMA_PROMPT},
        {"role": "user", "content": f"Question : {user_question}"}
    ]
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": 0  # Température 0 pour garantir la précision de la syntaxe
        }
    )
    
    cypher_query = response["message"]["content"].strip()
    
    # Nettoyage si le modèle ajoute malgré tout des balises markdown
    # if cypher_query.startswith("```cypher"):
    #     cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    # elif cypher_query.startswith("```"):
    #     cypher_query = cypher_query.replace("```", "").strip()
    match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", cypher_query, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return cypher_query

def generate_cypher_query(user_question: str, NEO4J_SCHEMA_PROMPT: str, model: str = "qwen2.5-coder:7b") -> str:
    messages = [
        {"role": "system", "content": NEO4J_SCHEMA_PROMPT},
        {"role": "user", "content": f"Question : {user_question}"}
    ]
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0}
    )
    
    cypher_query = response["message"]["content"].strip()
    
    # Capture le code à l'intérieur des blocs markdown ```cypher ... ``` ou ``` ... ```
    match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", cypher_query, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
        
    return cypher_query



def ask_graph(question: str, NEO4J_SCHEMA_PROMPT: str, driver, model="qwen2.5-coder:7b", display_query :bool = False):
    """_summary_

    Args:
        question (str): _description_
        NEO4J_SCHEMA_PROMPT (str): _description_
        driver (_type_): _description_
        model (str, optional): _description_. Defaults to "qwen2.5-coder:7b".

    Returns:
        _type_: _description_
    """
    # Create cypher query
    cypher_query = generate_cypher_query(question, NEO4J_SCHEMA_PROMPT, model=model)
    if display_query:
        print(f"\nCypher query\n{cypher_query}\n")
    
    # replace $query_embedding 
    params = {}
    if "$query_embedding" in cypher_query:
        # create the embedded vector
        params["query_embedding"] = text2vector(question)
    
    # Use the query
    with driver.session() as session:
        result = session.run(cypher_query, parameters=params)
        return [record.data() for record in result]

def ask_graph_app(question: str, NEO4J_SCHEMA_PROMPT: str, driver, model="qwen2.5-coder:7b", display_query: bool = False, return_raw_records: bool = False):
    # Création de la requête Cypher
    cypher_query = generate_cypher_query_app(question, NEO4J_SCHEMA_PROMPT, model=model)
    if display_query:
        print(f"\nCypher query:\n{cypher_query}\n\n")
    
    params = {}
    if "$query_embedding" in cypher_query:
        params["query_embedding"] = text2vector(question)
    
    with driver.session() as session:
        result = session.run(cypher_query, parameters=params)
        # Si on a besoin de construire le sous-graphe, on conserve les objets Neo4j bruts
        if return_raw_records:
            return [record for record in result]
        return [record.data() for record in result]

def answer_user(question: str, graph_results: list, model_llm :str = "qwen2.5:7b") -> str:
    prompt = f"""Tu es un assistant d'analyse politique.
À partir des données brutes Neo4j suivantes, réponds à la question de l'utilisateur.

Consignes :
1. Ignore les éléments de bruit (symboles, formules de politesse, fonctions génériques comme "Monsieur", "ministres").
2. Identifie les personnes réelles les plus pertinentes.
3. Sois concis et direct.

Question : {question}
Données Neo4j : {graph_results}
"""
    response = ollama.chat(
        model=model_llm,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    return response["message"]["content"].strip()


def answer_user_streamlit(question: str, graph_results: list) -> str:
    # Nettoyage et formatage strict
    cleaned_results = []
    for item in graph_results:
        filtered = {}
        for k, v in item.items():
            if isinstance(v, dict):
                filtered[k] = {p_k: p_v for p_k, p_v in v.items() if p_k != "embedding"}
            else:
                filtered[k] = v
        cleaned_results.append(filtered)

    prompt = f"""[INSTRUCTION CRITIQUE : RÉPONDS STRICTEMENT EN FRANÇAIS]
Tu es un assistant d'analyse politique. Analyse les données Neo4j ci-dessous pour répondre à la question.

RÈGLES D'INTERPRÉTATION IMPÉRATIVES :
1. Le champ "avg_score" ou "score" N'EST PAS un score de sentiment. C'est UNIQUEMENT un score de pertinence sémantique (similarité vectorielle). Ne parle JAMAIS de sentiment positif, négatif ou neutre.
2. Rédige l'intégralité de ta réponse EN FRANÇAIS.
3. Si des données semblent hors-sujet par rapport à la question, présente-les telles qu'elles apparaissent dans la base sans inventer de lien.

Question de l'utilisateur : {question}

Données Neo4j :
{cleaned_results}
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            # Forcer la langue dans le message système
            {"role": "system", "content": "Tu es un assistant francophone. Tu réponds TOUJOURS en français."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0}
    )
    return response["message"]["content"].strip()


def ask_graph_with_subgraph(question, prompt_schema, driver, llm_model="qwen2.5-coder:7b", debug=True):
    # Appel de ask_graph avec return_raw_records=True pour garder les objets Node/Relationship natifs
    raw_records = ask_graph_app(
        question=question,
        NEO4J_SCHEMA_PROMPT=prompt_schema,
        driver=driver,
        model=llm_model,
        display_query=debug,
        return_raw_records=True
    )
    
    if not isinstance(raw_records, list):
        return raw_records, {"nodes": [], "edges": []}

    nodes, edges = [], []
    seen_nodes = set()
    raw_data_formatted = []

    for record in raw_records:
        raw_data_formatted.append(record.data())
        for value in record.values():
            if value is None:
                continue
            
            # Traitement des Nœuds Neo4j (objets neo4j.graph.Node)
            if hasattr(value, "labels"):
                node_id = str(value.element_id if hasattr(value, "element_id") else value.id)
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "label": list(value.labels)[0] if value.labels else "Unknown",
                        "properties": dict(value)
                    })
            # Traitement des Relations Neo4j (objets neo4j.graph.Relationship)
            elif hasattr(value, "type"):
                edges.append({
                    "source": str(value.start_node.element_id if hasattr(value.start_node, "element_id") else value.start_node.id),
                    "target": str(value.end_node.element_id if hasattr(value.end_node, "element_id") else value.end_node.id),
                    "type": value.type
                })

    return raw_data_formatted, {"nodes": nodes, "edges": edges}