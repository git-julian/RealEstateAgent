import re
import os
import pandas as pd
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import json
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import shutil



# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API Key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY in the .env file.")


llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

def generate_listing(num_listings = 10):
    promt = f"""
    Your goal is to generate realistic real estate listings that includes the following parameters neighborhood, price, 
    bedroom count, bathroom count, house size, description of the house and neighborhood description.
    One Example is: Neighborhood: Green Oaks
    Price: $800,000
    Bedrooms: 3
    Bathrooms: 2
    House Size: 2,000 sqft

    Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

    Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
                
    Create {num_listings} listings.
    """  
    return(llm(promt))



def parse_real_estate_listings_to_json(input_str, json_filename='listings.json'):
    """
    Parses the given string containing real estate listings and stores the data in a JSON file.
    
    Parameters:
    - input_str (str): The raw string containing the real estate listings.
    - json_filename (str): The name of the output file. Default is 'listings.json'.
    
    Returns:
    - None
    """
    
    # Step 1: Split the string into individual real estate listings
    # Each listing starts with a number followed by '. Neighborhood:'
    property_split_pattern = re.compile(r'\n?\d+\.\s+Neighborhood:', re.MULTILINE)
    splits = property_split_pattern.split(input_str)
    
    # The first split is before the first property and is ignored
    properties = splits[1:]
    
    # Find all numbers for mapping (optional, if needed)
    numbers = re.findall(r'(\d+)\.\s+Neighborhood:', input_str)
    
    if len(numbers) != len(properties):
        print("Warning: The number of found numbers does not match the number of real estate listings.")
    
    # Define a regular expression to extract data from each property
    listing_pattern = re.compile(
        r'Neighborhood:\s*(?P<neighborhood>.+?)\n\s*'
        r'Price:\s*\$(?P<price>[\d,]+)\n\s*'
        r'Bedrooms:\s*(?P<bedrooms>\d+)\n\s*'
        r'Bathrooms:\s*(?P<bathrooms>[\d.]+)\n\s*'
        r'House Size:\s*(?P<house_size>[\d,]+)\s*sqft\n\s*'
        r'Description:\s*(?P<description>.+?)\n\s*'
        r'Neighborhood Description:\s*(?P<neighborhood_description>.+)', 
        re.DOTALL
    )
    
    listings = []
    
    for idx, prop in enumerate(properties, start=1):
        # Since the split removed "Neighborhood:", we add it back
        prop = 'Neighborhood:' + prop.strip()
        
        match = listing_pattern.search(prop)
        if match:
            data = match.groupdict()
            
            # Data Cleaning
            try:
                data['price'] = int(data['price'].replace(',', ''))
            except ValueError:
                print(f"Warning: Invalid price format in Property {idx}. Setting value as None.")
                data['price'] = None
            
            try:
                data['bedrooms'] = int(data['bedrooms'])
            except ValueError:
                print(f"Warning: Invalid bedrooms format in Property {idx}. Setting value as None.")
                data['bedrooms'] = None
            
            try:
                data['bathrooms'] = float(data['bathrooms'])
            except ValueError:
                print(f"Warning: Invalid bathrooms format in Property {idx}. Setting value as None.")
                data['bathrooms'] = None
            
            try:
                data['house_size'] = int(data['house_size'].replace(',', ''))
            except ValueError:
                print(f"Warning: Invalid house size format in Property {idx}. Setting value as None.")
                data['house_size'] = None
            
            # Remove leading and trailing whitespace from descriptions
            data['description'] = data['description'].strip()
            data['neighborhood_description'] = data['neighborhood_description'].strip()
            
            listings.append(data)
        else:
            print(f"Warning: Property {idx} does not match the expected format and was skipped.")
    
    # Save the data to a JSON file
    if listings:
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(listings, json_file, ensure_ascii=False, indent=4)
        print(f"Successfully saved {len(listings)} listings in '{json_filename}'.")
    else:
        print("No valid real estate listings found to save.")


def load_listings_data(file = 'listings.json'):

    if os.path.exists(file):
        df = pd.read_json(file)
    else:
        print(f"No listings data found. Please ensure {file} exists.")
    return df


def prepare_vectorstore(df):
    """
    Prepare the vector store for the real estate listings.

    - If a ChromaDB vector store exists in the 'chroma_db' directory, load it.
    - If not, create a new vector store from the provided DataFrame and persist it.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the listings data with a 'description' column.

    Returns:
    - vectorstore: An instance of the Chroma vector store.
    """
    if os.path.exists('chroma_db'):
        # Load the existing vector store
        shutil.rmtree('chroma_db')
        
  
    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Create Document objects with metadata
    docs = []
    for idx, row in df.iterrows():
        text = row['description']
        metadata = {
            "listing_id": idx,
            "neighborhood": row.get("neighborhood", ""),
            "price": row.get("price", ""),
            "bedrooms": row.get("bedrooms", ""),
            "bathrooms": row.get("bathrooms", ""),
            "house_size": row.get("house_size", ""),
            "source": f"listing_{idx}"
        }
        splits = text_splitter.split_text(text)
        for i, chunk in enumerate(splits):
            doc = Document(page_content=chunk, metadata=metadata)
            docs.append(doc)
            print(metadata)

        # Generate embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory='chroma_db'
        )
        vectorstore.persist()
        print("Created and persisted new ChromaDB vector store.")

    return vectorstore
