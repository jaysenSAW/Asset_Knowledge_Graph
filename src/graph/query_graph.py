import ollama
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

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



def generate_cypher_query(user_question: str, NEO4J_SCHEMA_PROMPT: str, model: str = "qwen2.5-coder:7b") -> str:
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
    if cypher_query.startswith("```cypher"):
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    elif cypher_query.startswith("```"):
        cypher_query = cypher_query.replace("```", "").strip()
        
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



def answer_user(question: str, graph_results: list) -> str:
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
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    return response["message"]["content"].strip()