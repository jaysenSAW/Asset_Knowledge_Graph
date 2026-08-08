import argparse
parser = argparse.ArgumentParser()


parser.add_argument('-f', '--folder', default='data/discours-presidents/', help='folder with speeches')
parser.add_argument('-llm', '--llm_model', default="qwen2.5:7b", help='llm model')
parser.add_argument('-p', '--prompt_template', default="src/prompts/prepare_llm_output_graph_production.json")
parser.add_argument('-r', '--prompt_RAG', default="src/prompts/prompt_RAG_production_V2.txt")
parser.add_argument('-n', '--neo4j_para', default="credential.json", help='USER, UI and PASSWORD for neo4j')
parser.add_argument('-log', '--log_path', default="output/log_graph.txt", help='log_graph.txt')
parser.add_argument('-o', '--output_folder', default="output", help='output_folder')
parser.add_argument('-g', '--global_report', default="output/global_report.json", help='output_folder')
parser.add_argument('-w', '--workers', type=int, default=4, help='number of parallel workers for LLM chunk extraction')
parser.add_argument('-s', '--save_every', type=int, default=1, help='write log/report to disk every N speeches (>1 speeds up I/O, less crash-safe)')



args = parser.parse_args()

from datetime import datetime
# import hashlib
import json
from neo4j import GraphDatabase
# import ollama
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.insert(1, "src/graph/")
from graph_builder import (
    get_node_id, 
    extract_graph, 
    compute_chunk_embeddings, 
    merge_graphs, 
    validate_graph, 
    build_neo4j_graph, 
    feed_global_report, 
    save_graph, 
    load_to_neo4j, 
    create_constraints
)
sys.path.insert(1, "src/preprocessing/")
from speeches import (
    load_speeches,
    split_into_chunks,
    load_prompt_template
)



folder = args.folder
#path_prompt_template = args.prompt_template
path_prompt_RAG = args.prompt_RAG
llm_model = args.llm_model
log_dic_path = args.log_path
output = args.output_folder
global_report_path = args.global_report
n_workers = args.workers
save_every = max(1, args.save_every)

MIN_CHUNK_LEN = 120

# Check if output folder exist
Path(output).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output, "merged_graph")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output, "neo4j_graph")).mkdir(parents=True, exist_ok=True)



# example_json = load_template_json(path_prompt_template)

# folder with the prompt used to extract informations
prompt_template = load_prompt_template(path_prompt_RAG)

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


# PROMPT_PREFIX = prompt_template.replace(
#     "{text}",
#     ""
# )

def extract_graph_for_chunk_old(chunk):

    prompt = PROMPT_PREFIX.replace(
        "{text}",
        chunk["text"]
    )

    try:
        return extract_graph(
            prompt,
            model=llm_model
        )
    except Exception as e:
        print(f"Chunk {chunk['chunk_id']} failed : {e}")
        return None

def extract_graph_for_chunk(chunk):

    prompt = prompt_template.replace(
        "{text}",
        chunk["text"]
    )

    print("=" * 100)
    print("PROMPT ENVOYÉ AU LLM")
    print("=" * 100)
    print(prompt)
    print("=" * 100)

    try:
        return extract_graph(
            prompt,
            model=llm_model
        )
    except Exception as e:
        print(f"Chunk {chunk['chunk_id']} failed : {e}")
        return None


def flush_state():
    """Persist log_dic and global_report to disk."""
    with open(log_dic_path, "w", encoding="utf-8") as f:
        json.dump(log_dic, f, indent=2, ensure_ascii=False)
    with open(global_report_path, "w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False)


# RUN PIPELINE

# extract data from speech and stores it into a list of dict with this fields :
# 'id', 'date', 'title', 'filename', 'header', 'text', 'chunks'
corpus = load_speeches(folder)

start_pipeline = time.time()

# Two dedicated, long-lived thread pools instead of creating/tearing down
# a ThreadPoolExecutor for every single speech (spawning threads 500 times
# adds up). llm_executor runs the (I/O-bound, waiting on Ollama) chunk
# extraction calls; embedding_executor runs chunk embeddings in the
# background so that work overlaps with the LLM calls instead of running
# strictly after them, since the two are independent (embeddings only need
# chunk text, not LLM output).
llm_executor = ThreadPoolExecutor(max_workers=n_workers)
embedding_executor = ThreadPoolExecutor(max_workers=1)

try:

    for i, speech in enumerate(tqdm(corpus[:1])):

        filename = speech["filename"]
        print("FILENAME")
        print(filename)
        # ----------------------------------------------------
        # Skip already processed files
        # ----------------------------------------------------

        if (
            filename in log_dic
            and log_dic[filename]["status"] == "SUCCESS"
            and log_dic[filename]["llm_model"] == llm_model
        ):
            continue

        speech_start = time.time()

        merged_graph = {
            "entities": [],
            "relations": []
        }

        try:

            ###################################################
            # 1. Chunking
            ###################################################

            speech["chunks"] = split_into_chunks(speech)

            ###################################################
            # 2. Kick off embeddings in the background, and start
            #    parallel LLM extraction at the same time. Both only
            #    depend on speech["chunks"] from step 1.
            ###################################################

            embedding_future = embedding_executor.submit(
                compute_chunk_embeddings,
                speech["chunks"]
            )

            graphs = []

            futures = [
                llm_executor.submit(extract_graph_for_chunk, chunk)
                for chunk in speech["chunks"]
                if len(chunk["text"]) > MIN_CHUNK_LEN
            ]

            for future in as_completed(futures):

                graph = future.result()

                if graph is not None:
                    graphs.append(graph)

            ###################################################
            # 3. Merge all graphs once
            ###################################################

            merged_graph = merge_graphs(graphs)

            ###################################################
            # 4. Validation
            ###################################################

            merged_graph, report = validate_graph(
                merged_graph
            )

            global_report = feed_global_report(
                report,
                speech,
                global_report
            )

            ###################################################
            # 5. Collect embeddings (while step 2 was waiting on
            #    the LLM)
            ###################################################

            speech["chunks"] = embedding_future.result()

            ###################################################
            # 6. Neo4j graph
            ###################################################

            neo4j_graph = build_neo4j_graph(
                speech,
                merged_graph
            )

            ###################################################
            # 7. Save JSON
            ###################################################

            save_graph(
                merged_graph,
                os.path.join(
                    output,
                    "merged_graph",
                    speech["id"] + ".json"
                )
            )

            save_graph(
                neo4j_graph,
                os.path.join(
                    output,
                    "neo4j_graph",
                    speech["id"] + ".json"
                )
            )

            ###################################################
            # 8. Load Neo4j
            ###################################################

            load_to_neo4j(
                driver,
                neo4j_graph
            )

            ###################################################
            # 9. Log SUCCESS
            ###################################################

            elapsed = time.time() - speech_start

            log_dic[filename] = {

                "status": "SUCCESS",

                "llm_model": llm_model,

                "prompt_RAG": path_prompt_RAG,

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

        except Exception as e:

            elapsed = time.time() - speech_start

            print(f"{filename} failed : {e}")

            log_dic[filename] = {

                "status": "FAILED",

                "llm_model": llm_model,

                "processed_at": datetime.now().isoformat(),

                "processing_time": round(elapsed, 2),

                "error": str(e)
            }
        # Save every N speeches
        if (i + 1) % save_every == 0:

            flush_state()

finally:

    flush_state()

    llm_executor.shutdown(wait=True)
    embedding_executor.shutdown(wait=True)

    driver.close()

print(
    f"\nPipeline finished in {time.time()-start_pipeline:.2f} sec"
)