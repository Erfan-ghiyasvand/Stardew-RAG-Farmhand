import json
import os
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return []
    
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)


def load_data_with_content_types(
    texts_path: str = None,
    tables_path: str = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Set default paths relative to the project root
    if texts_path is None:
        texts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "summarized_texts.json")
    if tables_path is None:
        tables_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "summarized_tables.json")
    # Load the data
    texts = load_json(texts_path)
    tables = load_json(tables_path)
    
    # Set content types
    for text in texts:
        text["content_type"] = "text"
    
    for table in tables:
        table["content_type"] = "table"
    
    print(f"SUCCESS: Loaded {len(texts)} text entries and {len(tables)} table entries")
    
    return texts, tables


def data_ingestion(
    texts_path: str = None,
    tables_path: str = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return load_data_with_content_types(texts_path, tables_path)


if __name__ == "__main__":
    # Example usage
    texts, tables = data_ingestion()
