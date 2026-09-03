import argparse
import json
import os
from glob import glob
from neo4j import GraphDatabase
from pathlib import Path
import sys
from tqdm import tqdm
sys.path.insert(1, "src/graph/")

from graph_builder import load_to_neo4j, create_constraints, check_database_empty


parser = argparse.ArgumentParser()

parser.add_argument('-c', '--neo4j_para', default="credential.json", help='USER, UI and PASSWORD for neo4j')
parser.add_argument('-g', '--neo4j_JSON', default="output/neo4j_graph", help='Folder name to get JSON')
parser.add_argument('-f', '--neo4j_constraint', default="src/preprocessing/graph_constraint.txt", help='Folder name to get JSON')

args = parser.parse_args()




# Connect to Neo4j
if os.path.exists(args.neo4j_para):
    with open(args.neo4j_para) as f:
        credential = json.load(f)
else:
    print("JSON not found for neo4j parameter")
    exit()

driver = GraphDatabase.driver(
    credential["URI"],
    auth=(credential["USER"], credential["PASSWORD"])
)



# Check database
if not check_database_empty(driver):
    print("Cancel :database is not empty. Delete it (MATCH (n) DETACH DELETE n;).")
    driver.close()
    exit()

print("Empty data base. Start INSERTION...")
# Add contraints and index
if os.path.exists(args.neo4j_para):
    create_constraints(driver, args.neo4j_constraint)
else:
    print("graph_constraint.txt not found")
    driver.close()
    exit()
# Load JSON from neo4j_graph
json_files = json_files = list(Path(args.neo4j_JSON).glob("*.json"))

if not json_files:
    print("No files found : {args.neo4j_JSON}")
else:
    print(f"Injection {len(json_files)}...")
    for filepath in tqdm(json_files, desc="Injecting graphs into Neo4j", unit="file"):
        with open(filepath, "r", encoding="utf-8") as f:
            graph = json.load(f)
            load_to_neo4j(driver, graph)
    print("Done !")

driver.close()