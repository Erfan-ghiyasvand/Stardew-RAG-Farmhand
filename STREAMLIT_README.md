# Stardew Valley RAG Streamlit App

A simple Streamlit interface for your Stardew Valley RAG pipeline.

## Prerequisites

1. **Qdrant Server**: Make sure your Qdrant server is running on `localhost:6333`
2. **OpenAI API Key**: Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Navigate to your project directory
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

1. Enter your question about Stardew Valley in the text area
2. Select your preferred AI model
3. Click "Get Answer" to get a response based on the Stardew Valley wiki

## Features

- Clean, user-friendly interface
- Model selection (GPT-4o-mini, GPT-4o, GPT-3.5-turbo)
- Error handling for connection issues
- Loading indicators during processing
- Responsive layout
