from qdrant_client import QdrantClient
from qdrant_client import models
from openai import OpenAI
import json
import uuid
import random 
from data_ingest import data_ingestion

OpenAIclient = OpenAI()


def llm(prompt, model='gpt-5-nano'):
    
    response = OpenAIclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def question_generation(knowledge_base , sampleNum = 10):
    
    question_generation_prompt = """
    You emulate a player of the Stardew Valley game.
    Here is the text from a wiki page of this game, along with the page and the section it was extracted from.
    Formulate a question that can be answered using these text materials.
    Only return the question. The questions should be complete and concise.
    Page title: {page_title}
    Section title: {section_title}
    Text: {text}\n
    """.strip()

    evaluation_questions = []

    sample_kb= random.sample(knowledge_base, sampleNum)

    for kb in sample_kb:
        eval = kb
        prompt = question_generation_prompt.format(page_title=kb["page_title"], section_title=kb["section_title"],text=kb["text"]).strip()
        question = llm(prompt)
        eval["question"] = question
        evaluation_questions.append(eval)

    return evaluation_questions   

def compute_mrr_and_hitrate(results, k=5):
    """
    results: list of lists of tuples (ranked_docs, correct_doc_id)
             e.g. [ (["(page1,sec1)", "(page2,sec2)", ...], "(page2,sec2)") , ... ]
    """
    reciprocal_ranks = []
    hits = 0

    for ranked_docs, correct_doc in results:
        # Find rank (1-indexed)
        rank = None
        for i, doc in enumerate(ranked_docs[:k]):
            if doc == correct_doc:
                rank = i + 1
                break

        if rank:
            reciprocal_ranks.append(1.0 / rank)
            hits += 1
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    hit_rate = hits / len(results)

    return mrr, hit_rate


def evaluate_search_functions(search_functions, k=5, sampleNum=5):
    knowledge_base, _ = data_ingestion()

    evaluation_dataset = question_generation(knowledge_base, sampleNum)
    all_results = {}

    for item in search_functions:
        # Handle both function and (name, function) tuple
        if isinstance(item, tuple):
            name, search_function = item
        else:
            search_function = item
            name = item.__name__

        print(f"\nEvaluating: {name}")
        results = []

        for dp in evaluation_dataset:
            query = dp["question"]
            correct_doc = (dp["page_title"], dp["section_title"])

            search_results = search_function(query=query)

            retrieved_ids = [
                (doc.payload["page_title"], doc.payload["section_title"])
                for doc in search_results
            ]

            results.append((retrieved_ids, correct_doc))

        mrr, hit_rate = compute_mrr_and_hitrate(results, k)
        print(f"{name} → MRR@{k}: {mrr:.3f}, HitRate@{k}: {hit_rate:.3f}")

        all_results[name] = {"MRR": mrr, "HitRate": hit_rate}

    return all_results
