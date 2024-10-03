from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
papi_key = os.getenv("PINECONE_API_KEY")
genai.configure(api_key=api_key)

# Pinecone setup
pc = Pinecone(api_key=papi_key)
index = pc.Index("god")

# Function to generate query embedding using Gemini Pro
def generate_query_embedding(query):
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",  # Gemini Pro embedding model
            content=query
        )
        # Extract and return the embedding from the response
        if 'embedding' in response:
            return response['embedding']
        else:
            raise ValueError("Failed to generate embeddings.")
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# Function to retrieve relevant chunks from Pinecone
def retrieve_relevant_chunks(query_embedding):
    try:
        query_response = index.query(
            vector=query_embedding,
            top_k=2,  # Adjust the number of top results you want
            include_metadata=True  # Return metadata (e.g., source information) along with vectors
        )
        # Extract relevant chunks from the Pinecone response
        return [match['metadata']['text'] for match in query_response['matches']]
    except Exception as e:
        st.error(f"Error retrieving chunks: {e}")
        return []

# Function to generate the final answer using Gemini Pro LLM
def generate_answer_with_gemini(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    
    # Crafting the prompt for GODAI
    prompt = f"""
    You are GODAI, an AI designed to provide comfort, wisdom, and guidance using the teachings from religious texts such as the Bible, Quran, Vedas, Torah, and other sacred writings.
    
    Context from sacred texts: {context}
    
    A user has asked the following question:
    "{query}"
    
    Based on the teachings from the sacred texts, provide a compassionate and insightful answer. Ensure your response is deeply rooted in religious wisdom, addressing not just the question but offering emotional support as well. Focus more on Hinduism and the Vedas.
    
    Answer:
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return None

# --- Streamlit App ---

# Set Streamlit page configuration
st.set_page_config(
    page_title="GOD AI - Spiritual Guidance",
    layout="centered",
    initial_sidebar_state="auto",
    page_icon="🌟"
)

# Apply custom CSS for styling and fixed footer
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle, #141E30, #243B55);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }

    h1 {
        text-align: center;
        font-size: 3.5em;
        color: #f39c12;
        font-weight: bold;
        margin-bottom: 20px;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px #000000;
    }

    .stButton button {
        background-color: #e74c3c;
        color: white;
        font-size: 1.2em;
        border-radius: 12px;
        padding: 10px 25px;
        margin-top: 20px;
        transition: background-color 0.3s ease;
    }

    .stButton button:hover {
        background-color: #c0392b;
    }

    .stTextInput input {
        font-size: 1.5em;
        padding: 10px;
        color: white;
        background-color: #34495e;
        border-radius: 12px;
    }

    /* Output text styling */
    .output-text {
        background-color: #2c3e50;
        padding: 20px;
        font-size: 1.4em; /* Set uniform font size for the output */
        line-height: 1.8;
        border-radius: 15px;
        color: #ecf0f1;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: white;
        background-color: #2c3e50;
        padding: 15px 0;
        font-size: 1.2em;
        letter-spacing: 1.5px;
        z-index: 1000; /* Ensure footer stays on top */
    }

    .description {
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 30px;
        color: #ecf0f1;
        line-height: 1.6;
        text-shadow: 1px 1px 3px #000000;
    }
    
    /* Starry background */
    .starry-sky {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('https://www.transparenttextures.com/patterns/stardust.png');
        z-index: -1;
        opacity: 0.3;
    }

    </style>
    """, unsafe_allow_html=True
)

# Starry background
st.markdown('<div class="starry-sky"></div>', unsafe_allow_html=True)

# Title and description
st.title("🌟 GOD AI - Spiritual Guidance 🌟")

# Deeper description
st.markdown(
    """
    <div class="description">
    GODAI is your celestial companion, drawing upon the eternal wisdom of sacred texts from across millennia.
    From the Bible, Quran, Vedas, Torah, and many more, GODAI offers timeless guidance to uplift your spirit and enrich your soul. 
    Seek answers from the universe, and let GODAI guide you toward comfort, clarity, and understanding.
    </div>
    """, unsafe_allow_html=True
)

# Input box for user query
query = st.text_input("Ask GODAI anything:")

# Button to submit the query
if st.button("Ask GOD AI"):
    if query:
        with st.spinner("GODAI is thinking..."):
            # Step 1: Embed the query using Gemini Pro
            query_embedding = generate_query_embedding(query)

            # Step 2: Retrieve relevant chunks from Pinecone based on query embedding
            retrieved_chunks = retrieve_relevant_chunks(query_embedding)

            # Step 3: Generate an answer using Gemini Pro LLM with the retrieved chunks
            answer = generate_answer_with_gemini(query, retrieved_chunks)

            # Display the answer in a styled box with uniform font size
            if answer:
                st.subheader("GODAI's Answer:")
                st.markdown(f'<div class="output-text">{answer}</div>', unsafe_allow_html=True)
            else:
                st.error("Could not generate an answer.")
    else:
        st.error("Please enter a question!")

# Footer with your name (Fixed footer)
st.markdown(
    """
    <footer>
        <div style="text-align:center;">
            <span>Made by Soham 🌟</span>
        </div>
    </footer>
    """, unsafe_allow_html=True
)