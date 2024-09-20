# 🏡 Real Estate Listings Project

This project provides a complete solution for generating and searching real estate listings using GPT-3.5 and a vector database. It is part of the Udacity Nano degree Generative AI. The task is to develop an innovative application named “HomeMatch”. This application leverages large language models (LLMs) and vector databases to transform standard real estate listings into personalized narratives that resonate with potential buyers’ unique preferences and needs. It utilizes OpenAI’s language model for creating synthetic listings and ChromaDB for storing and retrieving these listings based on user queries. For convenience, it also includes a Streamlit-based web application to interact with the listings more easily.

## Features

	•	Generate Synthetic Listings: Create realistic real estate listings using GPT-3.5.
	•	Semantic Search: Perform semantic searches over the listings using a vector database.
	•	Streamlit Web App: A user-friendly interface to interact with the generated listings and perform searches.

## Project Structure

	•	main.py: Main file for the Streamlit app

	•	listingcreation.py: Script containing functions for creating and processing listings

	•	listingsearch.py: Script containing search-related functions

	•	requirements.txt: List of required Python packages

	•	README.md: Project documentation (this file)

## Requirements

	•	Python 3.8 or higher
	•	OpenAI API key (Get it from OpenAI)
	•	The following Python packages (listed in requirements.txt):
	•	streamlit
	•	openai
	•	langchain
	•	chromadb
	•	pandas
	•	python-dotenv

Installation

	1.	Clone the Repository:
	•	git clone https://github.com/git-julian/RealEstateAgent/
	•	cd real-estate-listings
	2.	Install the Required Packages:
	•	pip install -r requirements.txt
	3.	Set Up the .env File:
	•	Create a .env file in the project root directory and add your OpenAI API key.

## Usage

Running the Streamlit App

To start the Streamlit web app, run:

	•	streamlit run main.py

## App Features

	1.	Generate New Listings:
		•	Choose the option to create new synthetic listings.
		•	Click on “Create Listings” to generate, parse, and store the listings in the vector database.
	2.	Search Existing Listings:
		•	Choose the option to use existing listings.
		•	Set your search preferences such as neighborhood, price range, bedrooms, etc.
		•	Enter additional requirements if needed.
		•	Click “Search” to perform a semantic search over the existing listings.
	3.	View Results:
		•	View the matching listings and their details.
		•	Generate a summary or exposé based on the search results.

## Functionality

### listingcreation.py

	•	generate_listing(num_listings=10): Generates synthetic real estate listings using GPT-3.5.
	•	parse_real_estate_listings_to_json(input_str, json_filename='listings.json'): Parses the generated listings and saves them to a JSON file.
	•	load_listings_data(file='listings.json'): Loads the listings data from the JSON file into a Pandas DataFrame.
	•	prepare_vectorstore(df): Prepares the ChromaDB vector store for storing the listings data.

### listingsearch.py

	•	perform_semantic_search(query, k=3): Performs a semantic search on the stored ChromaDB vector store.
	•	generate_summary(results): Generates a summary or exposé of the search results.