import argparse
parser = argparse.ArgumentParser()


parser.add_argument('-f', '--folder', default='data/discours-presidents/', help='folder with speeches')
parser.add_argument('-llm', '--llm_model', default="qwen2.5:7b", help='llm model')
parser.add_argument('-p', '--prompt_template', default="src/prompts/prepare_llm_output_graph_production.json")
parser.add_argument('-r', '--prompt_RAG', default="src/prompts/prompt_RAG_production.txt")
parser.add_argument('-n', '--neo4j_para', default="credential.json", help='USER, UI and PASSWORD for neo4j')
parser.add_argument('-log', '--log_path', default="output/log_graph.txt", help='log_graph.txt')
parser.add_argument('-o', '--output_folder', default="output", help='output_folder')
parser.add_argument('-g', '--global_report', default="output/global_report.json", help='output_folder')



args = parser.parse_args()

from datetime import datetime
# import hashlib
import json
from neo4j import GraphDatabase
# import ollama
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import time
from tqdm import tqdm
sys.path.insert(1, "src/graph/")
from graph_builder import get_node_id, extract_graph, compute_chunk_embeddings, merge_graphs, validate_graph, build_neo4j_graph, feed_global_report, save_graph, load_to_neo4j, create_constraints
sys.path.insert(1, "src/preprocessing/")
from speeches import load_speeches, split_into_chunks, load_template_json, load_prompt_template, prompt_builder



folder = args.folder
path_prompt_template = args.prompt_template
path_prompt_RAG = args.prompt_RAG
llm_model = args.llm_model
log_dic_path = args.log_path
output = args.output_folder
global_report_path = args.global_report

# Check if output folder exist
Path(output).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output, "merged_graph")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output, "neo4j_graph")).mkdir(parents=True, exist_ok=True)



example_json = load_template_json(path_prompt_template)

# folder with the prompt used to extract informations
prompt_template = load_prompt_template(path_prompt_RAG)

# Dict to save nodes and relationships
merged_graph = {
    "entities": [],
    "relations": []
}

if os.path.exists(global_report_path):
    with open(global_report_path) as json_file:
        global_report = json.load(json_file)
else:
    global_report = None

if os.path.exists(args.neo4j_para):
    with open(args.neo4j_para) as json_file:
        credential = json.load(json_file)
    print("Check connectivity with neo4j")
    print("need to lunch Neo4j desktop and DB before")
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
        os.path.join("src", "graph", "graph_constraint.txt")
    )
    print("Neo4j is connected !")

else:
    print("JSON not found for neo4j parameter")
    exit()


#Extraction pipeline
if os.path.exists(log_dic_path):
    with open(log_dic_path, "r", encoding="utf-8") as f:
        log_dic = json.load(f)
else:
    log_dic = {}


# RUN PIPELINE

# extract data from speech and sotres it into a list of dict with this fields : 
# 'id', 'date', 'title', 'filename', 'header', 'text', 'chunks'
corpus = load_speeches(folder)
start0 = time.time()
for speech in tqdm(corpus[:500]):#remove comment to lunch on all dataset
#for speech in corpus[:1]:
#check if the file was alreday present into the database
    filename = speech["filename"]
    if filename in log_dic.keys() and log_dic[filename]["status"] == "SUCCESS" and log_dic[filename]["llm_model"] == llm_model:
        #file already analyzed
        print(speech["filename"]+" already done")
        continue
    start = time.time()
    try:
        # split text in chuncks and store it
        speech["chunks"] = split_into_chunks(speech)
        # embedding
        speech["chunks"] = compute_chunk_embeddings(speech["chunks"])
        # label and relations for the speech
        merged_graph = {
            "entities": [],
            "relations": []
        }
        # split speech and made embedding
        for chunk in speech["chunks"]:
                prompt = prompt_builder(
                    prompt_template, 
                    example_json, 
                    chunk["text"]
                )
                try:
                    graph = extract_graph(
                        prompt, 
                        model = llm_model
                    )
                except Exception as e:
                    print(
                        f"Chunk {chunk['chunk_id']} failed : {e}"
                    )
                    continue
                if graph is None:
                    continue

                merged_graph = merge_graphs(
                    merged_graph,
                    graph
                )
        # Validation
        # check duplicate and no valide elements
        merged_graph, report = validate_graph(merged_graph)
        global_report = feed_global_report(
            report, 
            speech, 
            global_report
        )
        # build neo4j graph
        neo4j_graph = build_neo4j_graph(
            speech,
            merged_graph
        )
        save_graph(
            neo4j_graph,
            os.path.join(
                output, 
                "merged_graph",
                speech['id']+".json"
                )
        )
        save_graph(
            neo4j_graph,
            os.path.join(
                output, 
                "neo4j_graph",
                speech['id']+".json"
                )
        )

        # LOAD Neo4j
        load_to_neo4j(
            driver,
            neo4j_graph
        )
    except Exception as e:
        elapsed = time.time() - start
        # update log SUCCESS
        log_dic[filename] = {
            "status": "SUCCESS",
            "llm_model": llm_model,
            "prompt_version": "v1",
            "processed_at": datetime.now().isoformat(),
            "processing_time": round(elapsed, 2),
            "n_chunks": len(speech["chunks"]),
            "n_entities": len(
                merged_graph["entities"]
            ),
            "n_relations": len(
                merged_graph["relations"]
            )
        }
    finally:
        with open(log_dic_path, "w", encoding="utf-8") as f:
            json.dump(log_dic, f, indent=2, ensure_ascii=False)

    # Save the final report
    with open(global_report_path, "w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False)

    driver.close()
    print("\nPipeline done. Duration {0:.2f}".format(time.time() - start0))