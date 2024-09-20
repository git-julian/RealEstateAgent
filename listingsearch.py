import os
from chromadb.utils import embedding_functions
import pandas as pd
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Access the OpenAI API Key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY in the .env file.")


llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.5,openai_api_key=OPENAI_API_KEY)

def perform_semantic_search(query, k=3):
    """
    Perform a semantic search on the stored ChromaDB vector store.

    Parameters:
    - query (str): The search query string.
    - k (int): The number of top results to return. Default is 5.

    Returns:
    - results (list): A list of matching documents with their metadata.
    """
    # Ensure that the OpenAI API key is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Load the existing vector store
    if os.path.exists('chroma_db'):
        vectorstore = Chroma(
            persist_directory='chroma_db',
            embedding_function=OpenAIEmbeddings()
        )
    else:
        raise FileNotFoundError("ChromaDB not found. Please ensure that the vector store has been created and is located in the 'chroma_db' directory.")

    # Perform the similarity search
    search_results = vectorstore.similarity_search_with_score(query, k=k)

    # Prepare the results
    results = []
    for doc, score in search_results:
        result = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'score': score
        }
        results.append(result)

    return results



def generate_summary(results):
    """
    Generate a summary or exposé of the search results.

    Parameters:
    - results (list): List of dictionaries containing search results.

    Returns:
    - summary (str): A nicely formatted summary of the results.
    """
    # Ensure that the OpenAI API key is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")


    # Prepare the prompt for the LLM
    # Collect relevant information from results
    listings_info = []
    for result in results:
        metadata = result['metadata']
        listing_info = {
            'neighborhood': metadata.get('neighborhood', 'N/A'),
            'price': metadata.get('price', 'N/A'),
            'bedrooms': metadata.get('bedrooms', 'N/A'),
            'bathrooms': metadata.get('bathrooms', 'N/A'),
            'house_size': metadata.get('house_size', 'N/A'),
            'description': result['content']
        }
        listings_info.append(listing_info)

    # Create a prompt that asks the model to generate a summary
    prompt = "Based on the following real estate listings, write a compelling summary or exposé highlighting the key features and benefits of these properties:\n\n"
    for idx, info in enumerate(listings_info, 1):
        prompt += f"Listing {idx}:\n"
        prompt += f"Neighborhood: {info['neighborhood']}\n"
        prompt += f"Price: ${info['price']}\n"
        prompt += f"Bedrooms: {info['bedrooms']}\n"
        prompt += f"Bathrooms: {info['bathrooms']}\n"
        prompt += f"House Size: {info['house_size']} sqft\n"
        prompt += f"Description: {info['description']}\n\n"
    prompt += "Please provide a summary that would appeal to a potential buyer, focusing on how these listings meet the user's preferences."

    # Generate the summary
    try:
        summary = llm(prompt).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {e}")

    return summary

