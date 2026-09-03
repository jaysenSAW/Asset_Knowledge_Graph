import argparse
parser = argparse.ArgumentParser()


parser.add_argument('-f', '--folder', default='data/discours-presidents/', help='folder with speeches')
parser.add_argument('-llm', '--llm_cypher', default="qwen2.5-coder:7b", help='llm model')
parser.add_argument('-q', '--question', default="Combien de discours parle de la guerre en Ukraine ? Je veux une confidence de 0.6 au moins")
parser.add_argument('-n', '--neo4j_para', default="credential.json", help='USER, UI and PASSWORD for neo4j')
parser.add_argument('-s', '--schema_prompt_path', default="src/prompts/NEO4J_SCHEMA_PROMPT.txt", help='')
parser.add_argument('-c', '--chatbot_mode', default=True, help='Generate answer in natural language with an LLM')
parser.add_argument('-d', '--debug_mode', default=False, help='Dislay cypher query create by the LLM')



args = parser.parse_args()

import json
from neo4j import GraphDatabase
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import warnings
import logging
import warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*db.index.vector.queryNodes.*")


sys.path.insert(1, "src/graph/")
from query_graph import (
    create_vector_index,
    ask_graph,
    answer_user
)

from graph_builder import (
    create_constraints
)

NEO4J_SCHEMA_PROMPT_path = args.schema_prompt_path
question = args.question
llm_cypher = args.llm_cypher
debug_mode = args.debug_mode

# Check the presence of credential.json 
if os.path.exists("credential.json"):
    with open("credential.json") as json_file:
        credential = json.load(json_file)
    print("Check connectivity with neo4j")
    driver = GraphDatabase.driver(
        credential["URI"],
        auth=(
            credential["USER"],
            credential["PASSWORD"]
        )
    )
    driver.verify_connectivity()
    print("Load cypher script for constraint")
    create_constraints(
        driver,
        os.path.join("src", "preprocessing", "graph_constraint.txt")
    )
    print("Neo4j is connected !")
else:
    print("JSON not found for neo4j parameter")
    exit()

# In the graph database each chunk have the attribute `embedding`. 
# Before to compare our `query_embeddings`, we create vector index.
# The indexation step will sort the embedded vector in a way more effecient 
# for similarity comparison. The following query will create the index.
print("# create vector index")
create_vector_index(driver)
# Load NEO4J_SCHEMA_PROMPT
if os.path.exists(NEO4J_SCHEMA_PROMPT_path):
    with open(NEO4J_SCHEMA_PROMPT_path, "r") as filin:
        NEO4J_SCHEMA_PROMPT = filin.read()
else:
    print("Error no prompt schema found !")
    exit()


raw_result = ask_graph(question, NEO4J_SCHEMA_PROMPT, driver, llm_cypher, debug_mode)
if args.chatbot_mode:
    final_answer = answer_user(question, raw_result)
    print(f"\nRéponse finale :\n{final_answer}")
else:
    print(f"Réponse finale :\n{raw_result}")


