import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import re

# Matches a standalone "MOTS CLÉS" line (site footer with categorization
# keywords, an internal numeric id, and a "Haut de page" link) that some
# source pages append after the actual speech text. Everything from this
# marker onward is not part of the speech and must not be chunked / sent
# to the LLM, or it produces spurious entities (e.g. a "Theme" created
# from a keyword tag rather than an actual sentence in the speech).
FOOTER_MARKER_RE = re.compile(r"\n\s*mots[\s-]*cl[ée]s\s*\n", re.IGNORECASE)

# Matches a leading "Titre:" / "Titre :" label so the parsed title is just
# the title itself, not the raw label from the source file.
TITLE_LABEL_RE = re.compile(r"^titre\s*:\s*", re.IGNORECASE)

# Matches the "Intervenant(s) :" metadata label line in the header.
INTERVENANT_LABEL_RE = re.compile(r"^intervenant\(s\)\s*:?\s*$", re.IGNORECASE)

# Matches a generic "Some Label :" line, used to detect where the
# Intervenant(s) block ends (the next metadata label starts).
LABEL_LINE_RE = re.compile(r"^.+\s*:\s*$")


def extract_speakers(header: str) -> list:
    """Extract speaker name(s) from the 'Intervenant(s) :' metadata line
    in the speech header, if present.

    This is a deterministic, regex-based read of structured metadata
    already present in the source file — not an LLM inference from a
    pronoun like "je" in the speech body. The extraction prompt
    deliberately never resolves "je" to a named person to avoid
    hallucinating an identity; this sidesteps that limitation entirely by
    using a field the source explicitly labels, so it carries no risk of
    guessing wrong.

    Args:
        header (str): the speech's header (metadata before "Texte intégral")

    Returns:
        list[str]: speaker names found (empty list if none / not present)
    """
    lines = header.splitlines()
    speakers = []
    collecting = False

    for line in lines:
        stripped = line.strip()

        if collecting:
            if not stripped or LABEL_LINE_RE.match(stripped):
                break
            for name in stripped.split(","):
                name = name.strip()
                if name:
                    speakers.append(name)
            continue

        if INTERVENANT_LABEL_RE.match(stripped):
            collecting = True

    return speakers


def load_speeches(files_folder: str):
    """Load speech text files from a folder into a list of speech dicts.

    Files that are not regular files, that can't be decoded as UTF-8, or
    that don't contain the expected "Texte intégral" marker are skipped
    (with a warning) instead of crashing the whole loading step.

    Args:
        files_folder (str): folder containing the raw speech files

    Returns:
        list[dict]: one dict per successfully parsed speech
    """
    corpus = []
    for filename in sorted(os.listdir(files_folder)):
        filepath = os.path.join(files_folder, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except UnicodeDecodeError as e:
            print(f"Skipping {filename}: encoding error ({e})")
            continue

        # Split header / body
        try:
            header, body = raw_text.split("\nTexte intégral", maxsplit=1)
        except ValueError:
            print(f"Skipping {filename}: 'Texte intégral' marker not found")
            continue

        header_lines = header.splitlines()
        if not header_lines:
            print(f"Skipping {filename}: empty header, can't extract title")
            continue

        # First row = titre officiel (strip a leading "Titre:" label if present)
        title = header_lines[0].strip()
        title = TITLE_LABEL_RE.sub("", title).strip()

        # Speech body: strip a trailing "MOTS CLÉS" footer section if present,
        # since it's page navigation/categorization, not speech content.
        body = body.strip()
        footer_match = FOOTER_MARKER_RE.search(body)
        if footer_match:
            body = body[:footer_match.start()].strip()

        # Metadata
        speech = {
            "id": os.path.splitext(filename)[0],
            "date": filename.split("_")[0],
            "title": title,
            "filename": filename,
            "header": header,
            "text": body,
            "speakers": extract_speakers(header),
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
    with open(path, encoding="utf-8") as filin:
        return filin.read()



def prompt_builder(prompt_template, example_json, text):
    return prompt_template.format(
        example_json=example_json,
        text=text
    )