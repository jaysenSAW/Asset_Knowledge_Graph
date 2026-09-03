import json
import os
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase

# Imports internes issus de ton architecture
from src.graph.query_graph import create_vector_index, answer_user, ask_graph_with_subgraph
from src.graph.graph_builder import create_constraints

# Configuration de la page Streamlit
st.set_page_config(page_title="GraphRAG Presidential Speeches", layout="wide")
st.title("💬 GraphRAG — Discours Présidentiels")

# -------------------------------------------------------------------
# 1. Connexion & Initialisation (Mise en cache pour la légèreté)
# -------------------------------------------------------------------
@st.cache_resource
def init_neo4j_connection():
    if not os.path.exists("credential.json"):
        st.error("Fichier credential.json introuvable.")
        st.stop()
        
    with open("credential.json") as f:
        cred = json.load(f)
        
    driver = GraphDatabase.driver(cred["URI"], auth=(cred["USER"], cred["PASSWORD"]))
    driver.verify_connectivity()
    
    # Validation des contraintes & index vectoriel
    create_constraints(driver, os.path.join("src", "preprocessing", "graph_constraint.txt"))
    create_vector_index(driver)
    return driver

driver = init_neo4j_connection()

@st.cache_data
def load_prompt_schema():
    with open("src/prompts/NEO4J_SCHEMA_PROMPT.txt", "r", encoding="utf-8") as f:
        return f.read()

prompt_schema = load_prompt_schema()

# -------------------------------------------------------------------
# 2. Interface Utilisateur
# -------------------------------------------------------------------
question = st.text_input("Posez votre question :", value="Montre-moi les 5 morceaux de discours qui parlent de la transition écologique ainsi que les entités mentionnées.")

if st.button("Rechercher", type="primary"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        with st.spinner("Analyse du graphe et génération de la réponse..."):
            # Exécution du pipeline GraphRAG
            raw_result, subgraph = ask_graph_with_subgraph(
                question, prompt_schema, driver, llm_model="qwen2.5-coder:7b"
            )
            final_answer = answer_user(question, raw_result)

        # Diviser la vue en deux colonnes
        col_text, col_graph = st.columns([1, 1])

        # Colonne Gauche : Réponse en langage naturel
        with col_text:
            st.subheader("💡 Réponse")
            st.write(final_answer)
            
            with st.expander("Voir les données brutes renvoyées par Cypher"):
                st.json(raw_result)

        # Colonne Droite : Visualisation du sous-graphe
        with col_graph:
            st.subheader("🕸️ Sous-graphe source")
            
            if not subgraph["nodes"]:
                st.info("Aucun nœud à afficher pour cette requête.")
            else:
                agraph_nodes = []
                agraph_edges = []

                # Couleurs personnalisées par Label
                color_map = {
                    "Speech": "#FF6B6B",
                    "Chunk": "#4D96FF",
                    "Person": "#6BCB77",
                    "Organization": "#FFD93D",
                    "Country": "#9B59B6"
                }

                # Construction des nœuds graphiques
                for n in subgraph["nodes"]:
                    label_type = n["label"]
                    props = n["properties"]
                    
                    # Titre affiché sur le nœud
                    display_name = props.get("title") or props.get("name") or f"Chunk_{props.get('chunk_number', '')}"
                    
                    agraph_nodes.append(Node(
                        id=n["id"],
                        label=str(display_name)[:20],
                        size=25 if label_type == "Chunk" else 30,
                        color=color_map.get(label_type, "#AAAAAA"),
                        title=f"<b>{label_type}</b><br>{json.dumps(props, ensure_ascii=False, indent=2)}"
                    ))

                # Construction des relations
                for e in subgraph["edges"]:
                    agraph_edges.append(Edge(
                        source=e["source"],
                        target=e["target"],
                        label=e["type"],
                        color="#CCCCCC"
                    ))

                # Configuration du rendu graphique (Léger & dynamique)
                config = Config(
                    width=650,
                    height=500,
                    directed=True,
                    physics=True,
                    hierarchical=False
                )

                # Rendu du sous-graphe
                agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)