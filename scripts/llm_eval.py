from RAG_pipeline import rag, llm

def llm_eval(models, query):
    """
    Simple LLM evaluation function that compares answers from different models.
    
    Args:
        models: List of model names to test
        query: Question to ask all models
    
    Returns:
        Dictionary with model answers and judge's evaluation
    """
    # Get answers from all models
    answers = {}
    for model in models:
        answers[model] = rag(query, model=model)
    
    # Create judge prompt
    judge_prompt = f"""
    You are an expert judge evaluating AI model responses. 
    
    Question: {query}
    
    Model Answers:
    """
    
    for model, answer in answers.items():
        judge_prompt += f"\n{model}: {answer}\n"
    
    judge_prompt += """
    
    Please evaluate these answers and provide:
    1. Which answer is best and why
    2. Brief scores (1-10) for each model
    3. Any notable differences
    
    Keep your evaluation concise and objective.
    """
    
    # Get judge's evaluation
    evaluation = llm(judge_prompt)
    
    return {
        "query": query,
        "answers": answers,
        "evaluation": evaluation
    }
