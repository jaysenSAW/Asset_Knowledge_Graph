import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def load_speeches(files_folder : str):
    """_summary_

    Args:
        files_folder (str): _description_

    Returns:
        _type_: _description_
    """
    corpus = []
    for filename in os.listdir(files_folder):
        #read the files
        with open(os.path.join(files_folder, filename), "r", encoding="utf-8") as f:
            raw_text = f.read()
        # Split hearder
        header, body = raw_text.split("\nTexte intégral", maxsplit=1)

        # First row = titre officiel
        title = header.splitlines()[0].strip()

        # Speech
        body = body.strip()

        # Metadata
        speech = {
            "id": os.path.splitext(filename)[0],
            "date": filename.split("_")[0],
            "title": title,
            "filename": filename,
            "header": header,
            "text": body,
            "chunks": []
        }
        corpus.append(speech)
    return corpus

def split_into_chunks(speech, chunk_size = 1200, chunk_overlap=200):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " "
        ]
    )

    texts = splitter.split_text(speech["text"])

    return [
        {
            "speech_id": speech["id"],
            "chunk_id": i,
            "date": speech["date"],
            "title": speech["title"],
            "text": chunk
        }
        for i, chunk in enumerate(texts)
    ]



def load_template_json(path : str):
    with open(
        path,
        encoding="utf-8"
    ) as json_file:
        example_json = json.load(json_file)

    return json.dumps(
        example_json,
        ensure_ascii=False,
        indent=2
    )

def load_prompt_template(path):
    prompt_template = ""
    with open(path, encoding="utf-8") as filin:
        for line in filin:
            prompt_template += line
    return prompt_template



def prompt_builder(prompt_template, example_json, text):
    return prompt_template.format(
        example_json=example_json,
        text=text
    )