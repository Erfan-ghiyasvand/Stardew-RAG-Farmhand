from qdrant_client import QdrantClient
from qdrant_client import models
from openai import OpenAI


qdClient = QdrantClient("http://localhost:6333")
OpenAIclient = OpenAI()

collection_name="stardew-sparse-and-dense"
vector_model_handle = "jinaai/jina-embeddings-v2-small-en"
EMBEDDING_DIMENSIONALITY = 512
spasrse_model_handle="Qdrant/bm25"


def multi_stage_search(query ,client=qdClient, collection_name=collection_name,limit= 5):
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=vector_model_handle,
                ),
                using="jina-small",
                # Prefetch three times more results, then
                # expected to return, so we can really rerank
                limit=(3 * limit),
            ),
        ],
        query=models.Document(
            text=query,
            model=spasrse_model_handle, 
        ),
        using="bm25",
        limit=limit,
        with_payload=True,
    )

    return results.points


def rrf_search(query,client =qdClient, collection_name = collection_name , limit = 5):
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=vector_model_handle,
                ),
                using="jina-small",
                limit=(5 * limit),
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=spasrse_model_handle,
                ),
                using="bm25",
                limit=(5 * limit),
            ),
        ],
        # Fusion query enables fusion on the prefetched results
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )

    return results.points[:limit]


def build_prompt(question, search_results):
    prompt_template = """
    You're an AI assistant for the players of a computer game named Stardew Valley.
    Answer the QUESTION based on the CONTEXT extracted from the game's wiki website. Some materials are texts and some are html tables.
    Use only the materials from the CONTEXT when answering the QUESTION, and don't use your own knowledge of the game.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        if (doc.payload["content_type"] =="text"):
            context += (
            f"Page title: {doc.payload['page_title']}\n"
            f"Section title: {doc.payload['section_title']}\n"
            f"Text: {doc.payload['text']}\n\n\n"
            )
        elif (doc.payload["content_type"] =="table"):
            context += (
            f"Page title: {doc.payload['page_title']}\n"
            f"Section title: {doc.payload['section_title']}\n"
            f"Table HTML: {doc.payload['table_html']}\n\n\n"
            )

    prompt = prompt_template.format(question=question, context=context).strip()
    return prompt


def llm(prompt, model='gpt-5-mini'):
    response = OpenAIclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def rag(query, model='gpt-5-mini'):
    search_results = rrf_search(client=qdClient,collection_name=collection_name,query=query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer