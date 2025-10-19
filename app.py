import streamlit as st
import sys
import os

# Add the scripts directory to the path so we can import RAG_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from RAG_pipeline import rag, multi_stage_search, rrf_search
from llm_eval import llm_eval
from Retrieval_evaluation import evaluate_search_functions

# Configure the page
st.set_page_config(
    page_title="Stardew Valley RAG Assistant",
    page_icon="🌾",
    layout="wide"
)

# Title and description
st.title("🌾 Stardew Valley RAG Assistant")
st.markdown("Ask questions about Stardew Valley and get answers based on the game's wiki!")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Single Model Query", "Model Comparison", "Retrieval Evaluation"])

with tab1:
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Ask a Question")
        
        # Text input for the query
        query = st.text_area(
            "Enter your question about Stardew Valley:",
            placeholder="e.g., How do I get iridium ore? What crops should I plant in spring?",
            height=100
        )
        
        # Model selection
        model = st.selectbox(
            "Select AI Model:",
            ["gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4o"],
            index=0
        )
            
        # Submit button
        submit_button = st.button("Get Answer", type="primary")

    with col2:
        st.header("Answer")
        
        if submit_button and query:
            with st.spinner("Searching the Stardew Valley wiki and generating answer..."):
                try:
                    # Call the RAG pipeline
                    answer = rag(query, model=model)
                    
                    # Display the answer
                    st.success("Answer generated successfully!")
                    st.markdown("### Response:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Make sure your Qdrant server is running on localhost:6333 and your OpenAI API key is set.")
        
        elif submit_button and not query:
            st.warning("Please enter a question first!")
        
        else:
            st.info("Enter a question and click 'Get Answer' to get started!")

with tab2:
    st.header("🤖 Model Comparison")
    st.markdown("Compare answers from multiple AI models using an LLM judge.")
    
    # Create two columns for comparison
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        # Query input for comparison
        eval_query = st.text_area(
            "Enter your question for model comparison:",
            placeholder="e.g., How do I get iridium ore? What crops should I plant in spring?",
            height=100,
            key="eval_query"
        )
        
        # Model selection for comparison
        available_models = ["gpt-5","gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4o"]
        selected_models = st.multiselect(
            "Select models to compare:",
            available_models,
            default=["gpt-4o", "gpt-4o-mini"],
            key="selected_models"
        )
        
        # Judge model selection
        judge_model = st.selectbox(
            "Select judge model:",
            available_models,
            index=2,  # Default to gpt-4o-mini
            key="judge_model"
        )
        
        # Submit button for comparison
        compare_button = st.button("Compare Models", type="primary", key="compare_button")
    
    with col2:
        st.subheader("Results")
        
        if compare_button and eval_query and selected_models:
            with st.spinner("Running model comparison and evaluation..."):
                try:
                    # Run the evaluation
                    result = llm_eval(selected_models, eval_query)
                    
                    # Display results
                    st.success("Model comparison completed!")
                    
                    # Show individual answers
                    st.markdown("### Model Answers:")
                    for model, answer in result["answers"].items():
                        with st.expander(f"**{model}**"):
                            st.write(answer)
                    
                    # Show judge evaluation
                    st.markdown("### Judge Evaluation:")
                    st.info(result["evaluation"])
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Make sure your Qdrant server is running on localhost:6333 and your OpenAI API key is set.")
        
        elif compare_button and not eval_query:
            st.warning("Please enter a question for comparison!")
        
        elif compare_button and not selected_models:
            st.warning("Please select at least one model to compare!")
        
        else:
            st.info("Enter a question, select models, and click 'Compare Models' to get started!")

with tab3:
    st.header("🔍 Retrieval Evaluation")
    st.markdown("Evaluate different search strategies using MRR (Mean Reciprocal Rank) and Hit Rate metrics.")
    
    # Create two columns for evaluation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        # Search function selection
        search_functions = {
            "RRF Search": ("rrf_search", rrf_search),
            "Multi-stage Search": ("multi_stage_search", multi_stage_search)
        }
        
        selected_functions = st.multiselect(
            "Select search functions to evaluate:",
            list(search_functions.keys()),
            default=["RRF Search"],
            key="selected_search_functions"
        )
        
        # Evaluation parameters
        st.subheader("Evaluation Parameters")
        
        sample_size = st.number_input(
            "Number of questions to generate:",
            min_value=1,
            max_value=50,
            value=5,
            help="More questions provide more reliable results but take longer to process"
        )
        
        k_value = st.number_input(
            "Top-K for evaluation:",
            min_value=1,
            max_value=20,
            value=5,
            help="Evaluate retrieval performance in top-K results"
        )
        
        # Submit button for evaluation
        eval_button = st.button("Run Retrieval Evaluation", type="primary", key="eval_button")
    
    with col2:
        st.subheader("Results")
        
        if eval_button and selected_functions:
            with st.spinner("Running retrieval evaluation... This may take a few minutes."):
                try:
                    # Prepare search functions for evaluation
                    functions_to_eval = []
                    for func_name in selected_functions:
                        name, func = search_functions[func_name]
                        functions_to_eval.append((name, func))
                    
                    # Run evaluation
                    results = evaluate_search_functions(
                        functions_to_eval, 
                        k=k_value, 
                        sampleNum=sample_size
                    )
                    
                    # Check if evaluation failed
                    if not results:
                        st.error("Evaluation failed - no data available or error occurred. Check the console for details.")
                        st.stop()
                    
                    # Display results
                    st.success("Retrieval evaluation completed!")
                    
                    # Create metrics display
                    st.markdown("### Evaluation Results:")
                    
                    # Create a results table
                    import pandas as pd
                    
                    results_data = []
                    for func_name, metrics in results.items():
                        results_data.append({
                            "Search Function": func_name,
                            f"MRR@{k_value}": f"{metrics['MRR']:.3f}",
                            f"Hit Rate@{k_value}": f"{metrics['HitRate']:.3f}"
                        })
                    
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Add some interpretation
                    st.markdown("### Interpretation:")
                    st.info("""
                    - **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of the first relevant document
                    - **Hit Rate**: Percentage of queries where at least one relevant document was found in top-K
                    - Higher values indicate better retrieval performance
                    """)
                    
                    # Find best performing function
                    best_mrr = max(results.items(), key=lambda x: x[1]['MRR'])
                    best_hitrate = max(results.items(), key=lambda x: x[1]['HitRate'])
                    
                    col_mrr, col_hit = st.columns(2)
                    with col_mrr:
                        st.metric(
                            "Best MRR", 
                            f"{best_mrr[1]['MRR']:.3f}",
                            f"{best_mrr[0]}"
                        )
                    with col_hit:
                        st.metric(
                            "Best Hit Rate", 
                            f"{best_hitrate[1]['HitRate']:.3f}",
                            f"{best_hitrate[0]}"
                        )
                    
                except Exception as e:
                    st.error(f"An error occurred during evaluation: {str(e)}")
                    st.info("Make sure your Qdrant server is running on localhost:6333 and your OpenAI API key is set.")
        
        elif eval_button and not selected_functions:
            st.warning("Please select at least one search function to evaluate!")
        
        else:
            st.info("Select search functions and click 'Run Retrieval Evaluation' to get started!")

# Footer
st.markdown("---")
st.markdown("**Note:** This assistant uses the Stardew Valley wiki as its knowledge base. Make sure your Qdrant vector database is running and contains the processed wiki data.")
