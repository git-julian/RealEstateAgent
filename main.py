
import streamlit as st
import os
from listingcreation import generate_listing, parse_real_estate_listings_to_json, load_listings_data, prepare_vectorstore
from listingsearch import perform_semantic_search
from langchain.llms import OpenAI



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

    # Initialize the OpenAI LLM
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

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

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Real Estate Assistant", page_icon=":house:", layout="wide")
    
    st.title("Real Estate Assistant :house:")
    st.markdown("Welcome! You can either generate new synthetic real estate listings or use existing data to find your ideal home.")
    
    # Sidebar for options
    st.sidebar.header("Options")
    user_choice = st.sidebar.radio("Choose an option:", ("Generate New Listings", "Use Existing Listings"))
    
    if user_choice == "Generate New Listings":
        st.header("Generate New Synthetic Listings")
        if st.button("Create Listings"):
            with st.spinner("Generating synthetic listings..."):
                try:
                    # Step 1: Generate listings
                    listings = generate_listing()
                    st.success("Synthetic listings generated successfully!")
                    
                    # Step 2: Parse listings to JSON
                    parse_real_estate_listings_to_json(listings)
                    st.success("Listings parsed to JSON.")
                    
                    # Step 3: Load listings data
                    df = load_listings_data()
                    st.success("Listings data loaded.")
                    
                    # Step 4: Prepare vector store
                    vectorstore = prepare_vectorstore(df)
                    st.success("Vector store created and ready for searches.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    elif user_choice == "Use Existing Listings":
        st.header("Search Existing Listings")
        # Check if vector store exists
        if not os.path.exists('chroma_db'):
            st.error("No existing vector store found. Please generate new listings first.")
        else:
            # User preferences
            st.subheader("Select Your Preferences")
            # Example options - customize as per your data
            neighborhood = st.multiselect("Preferred Neighborhoods:", options=["Green Oaks", "Sunnybrook", "Maple Ridge", "Harbor View", "Cedar Hills", "Lakeside Estates", "Willow Springs", "Riverstone", "Mountain View", "Pine Grove"])
            price_range = st.slider("Price Range ($):", 100000, 2000000, (300000, 800000), step=50000)
            bedrooms = st.multiselect("Number of Bedrooms:", options=[2, 3, 4, 5], default=[3])
            bathrooms = st.multiselect("Number of Bathrooms:", options=[1.5, 2, 2.5, 3, 3.5, 4], default=[2])
            house_size = st.slider("House Size (sqft):", 1000, 5000, (1500, 3000), step=100)
            
            # Add a text field for individual tastes
            additional_requirements = st.text_area("Additional Requirements or Preferences:")
            
            if st.button("Search"):
                with st.spinner("Formulating your search query..."):
                    # Formulate search query based on preferences
                    query_parts = []
                    if neighborhood:
                        query_parts.append(f"in {', '.join(neighborhood)}")
                    if price_range:
                        query_parts.append(f"priced between ${price_range[0]:,} and ${price_range[1]:,}")
                    if bedrooms:
                        bed_str = ' or '.join([f"{b}" for b in bedrooms])
                        query_parts.append(f"having {bed_str} bedrooms")
                    if bathrooms:
                        bath_str = ' or '.join([f"{b}" for b in bathrooms])
                        query_parts.append(f"having {bath_str} bathrooms")
                    if house_size:
                        query_parts.append(f"with house size between {house_size[0]} and {house_size[1]} sqft")
                    if additional_requirements:
                        query_parts.append(f"that {additional_requirements}")
                    
                    query = ", ".join(query_parts) + "."
                    st.write(f"**Search Query:** {query}")
                
                with st.spinner("Performing semantic search..."):
                    try:
                        results = perform_semantic_search(query)
                        
                        if results:

                            with st.spinner("Generating summary..."):
                                try:
                                    summary = generate_summary(results)
                                    st.header("Summary of Listings")
                                    st.write(summary)
                                except Exception as e:
                                    st.error(f"An error occurred while generating the summary: {e}")

                                    
                            st.success(f"Found {len(results)} matching listings:")
                            for idx, result in enumerate(results, 1):
                                st.markdown(f"### Listing {idx}")
                                st.markdown(f"**Score:** {result['score']:.4f}")
                                st.markdown(f"**Listing ID:** {result['metadata'].get('listing_id', 'N/A')}")
                                st.markdown(f"**Neighborhood:** {result['metadata'].get('neighborhood', 'N/A')}")
                                st.markdown(f"**Price:** ${result['metadata'].get('price', 'N/A'):,}")
                                st.markdown(f"**Bedrooms:** {result['metadata'].get('bedrooms', 'N/A')}")
                                st.markdown(f"**Bathrooms:** {result['metadata'].get('bathrooms', 'N/A')}")
                                st.markdown(f"**House Size:** {result['metadata'].get('house_size', 'N/A')} sqft")
                                st.markdown(f"**Description:** {result['content']}")
                                st.markdown("---")
                            
                            # Generate and display the summary
                            
                        else:
                            st.info("No matching listings found based on your preferences.")
                    
                    except Exception as e:
                        st.error(f"An error occurred during the search: {e}")

if __name__ == '__main__':
    main()