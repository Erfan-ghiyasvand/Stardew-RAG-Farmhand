from qdrant_client import QdrantClient
from qdrant_client import models
import uuid
from typing import List, Dict, Any
from data_ingest import data_ingestion


def create_collection(
    qdClient: QdrantClient,
    collection_name: str,
    embedding_dimensionality: int = 512
):
    """
    Create a Qdrant collection with dense and sparse vectors.
    
    Args:
        qdClient: Qdrant client instance
        collection_name: Name of the collection to create
        embedding_dimensionality: Size of the dense vectors
    """
    qdClient.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Named dense vector for jinaai/jina-embeddings-v2-small-en
            "jina-small": models.VectorParams(
                size=embedding_dimensionality,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )


def build_points(
    texts: List[Dict[str, Any]], 
    tables: List[Dict[str, Any]], 
    vector_model_handle: str, 
    sparse_model_handle: str, 
    embedding_dim: int = 512
) -> List[models.PointStruct]:
    """
    Build Qdrant points from texts and tables data.
    
    Args:
        texts: List of text documents
        tables: List of table documents
        vector_model_handle: Model handle for dense vectors
        sparse_model_handle: Model handle for sparse vectors
        embedding_dim: Embedding dimensionality
        
    Returns:
        List of PointStruct objects for Qdrant
    """
    points = []
    
    for text in texts:
        text_to_embedd = f"Page title (2X importance): {text.get('page_title','')}. Section title: {text.get('section_title','')}. text: {text.get('text','')}"
        point = models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "jina-small": models.Document(
                    text=text_to_embedd,
                    model=vector_model_handle,
                ),
                "bm25": models.Document(
                    text=text_to_embedd,
                    model=sparse_model_handle,
                ),
            },
            payload=text
        )
        points.append(point)

    for table in tables:
        text_to_embedd = f"Page title (2X importance): {table.get('page_title','')}. Section title: {table.get('section_title','')}. Table summary: {table.get('summary','')}"
        point = models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "jina-small": models.Document(
                    text=text_to_embedd,
                    model=vector_model_handle,
                ),
                "bm25": models.Document(
                    text=text_to_embedd,
                    model=sparse_model_handle,
                ),
            },
            payload=table
        )
        points.append(point)

    return points


def batch_upsert(
    qdClient: QdrantClient, 
    collection_name: str, 
    points: List[models.PointStruct], 
    batch_size: int = 500
):
    """
    Upsert points to Qdrant in batches.
    
    Args:
        qdClient: Qdrant client instance
        collection_name: Name of the collection
        points: List of points to upsert
        batch_size: Size of each batch
    """
    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i+batch_size]
        qdClient.upsert(collection_name=collection_name, points=batch)
        print(f"SUCCESS: Upserted {min(i+batch_size, total)}/{total}")


def vector_store_pipeline(
    url: str = "http://localhost:6333",
    collection_name: str = "stardew-sparse-and-dense",
    vector_model_handle: str = "jinaai/jina-embeddings-v2-small-en",
    embedding_dimensionality: int = 512,
    sparse_model_handle: str = "Qdrant/bm25",
    texts_path: str = "data/summarized_texts.json",
    tables_path: str = "data/summarized_tables.json",
    batch_size: int = 1000
):
    """
    Complete vector store pipeline using URL parameter.
    
    Args:
        url: Qdrant server URL 
        collection_name: Name of the collection
        vector_model_handle: Model handle for dense vectors
        embedding_dimensionality: Size of dense vectors
        sparse_model_handle: Model handle for sparse vectors
        texts_path: Path to texts JSON file
        tables_path: Path to tables JSON file
        batch_size: Batch size for upserting
        
    Returns:
        QdrantClient instance and collection name
    """
    # Initialize Qdrant client with URL
    qdClient = QdrantClient(url=url)
    
    # Create collection
    create_collection(qdClient, collection_name, embedding_dimensionality)
    
    # Load data with content types
    texts, tables = data_ingestion(texts_path, tables_path)
    
    # Build points
    points = build_points(texts, tables, vector_model_handle, sparse_model_handle, embedding_dimensionality)
    
    # Batch upsert
    batch_upsert(qdClient, collection_name, points, batch_size)
    
    print(f"SUCCESS: Ingested {len(points)} points into collection '{collection_name}'")
    
    return qdClient, collection_name


def main():
    """
    Main function to run the vector store pipeline.
    """
    vector_store_pipeline()


if __name__ == "__main__":
    main()

