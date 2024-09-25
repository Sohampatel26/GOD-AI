import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
extracted_data=extract_text_from_pdf('/Users/soham/Documents/GitHub/GOD-AI/Data/Sacred.pdf')

# Function to split the extracted text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks
splitted_chunks=split_text_into_chunks(extracted_data)

def generate_gemini_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        if chunk.strip():  # Ensure the chunk is not empty
            response = genai.embed_content(
                model="models/text-embedding-004",  # Gemini Pro embedding model
                content=chunk
            )
            
            # Now we directly access 'embedding' as it contains the values directly
            if isinstance(response, dict) and 'embedding' in response:
                embeddings.append(response['embedding'])  # Append the embedding directly
    return embeddings
embedded_chunks=generate_gemini_embeddings(splitted_chunks)

load_dotenv()
papi_key = os.getenv("PINECONE_API_KEY")
print(papi_key)
pc = Pinecone(api_key=papi_key)
index = pc.Index("god")

def upsert_embeddings_in_batches(text_chunks, embeddings, batch_size=100):
    vectors = []
    
    for i, embedding in enumerate(embeddings):
        # Create metadata for each chunk
        metadata = {"text": text_chunks[i], "source": "your_document_source"}
        vector = {
            "id": f"vec{i}",  # Unique ID for each vector
            "values": embedding,  # The embedding values
            "metadata": metadata  # Metadata for the chunk
        }
        vectors.append(vector)
        
        # Batch upsert every `batch_size` chunks
        if (i + 1) % batch_size == 0 or (i + 1) == len(embeddings):
            index.upsert(vectors=vectors, namespace="ns1")
            vectors = []  # Clear the list after each batch

# Call the function to upsert embeddings in batches
upsert_embeddings_in_batches(splitted_chunks, embedded_chunks, batch_size=100)