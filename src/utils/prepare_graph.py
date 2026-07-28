import unicodedata
import hashlib

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
    name = unicodedata.normalize(
        "NFKD",
        name
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