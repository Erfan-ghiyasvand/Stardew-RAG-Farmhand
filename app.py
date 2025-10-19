import streamlit as st
import sys
import os

# Add the scripts directory to the path so we can import RAG_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from RAG_pipeline import rag
from llm_eval import llm_eval

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
tab1, tab2 = st.tabs(["Single Model Query", "Model Comparison"])

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

# Footer
st.markdown("---")
st.markdown("**Note:** This assistant uses the Stardew Valley wiki as its knowledge base. Make sure your Qdrant vector database is running and contains the processed wiki data.")
