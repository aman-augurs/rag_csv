import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st

# Load SentenceTransformer for embeddings
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    generator = pipeline('text2text-generation', model='t5-small')
    return embedder, generator

# Function to load and preprocess CSV
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    df['content'] = df.astype(str).agg(' '.join, axis=1)
    return df

# Build FAISS index
def build_faiss_index(data, embedder):
    embeddings = embedder.encode(data['content'].tolist(), convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Search function for FAISS
def search_index(query, index, data, embedder, top_k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Ensure top_k doesn't exceed the number of available entries
    top_k = min(top_k, len(data))
    
    distances, indices = index.search(query_embedding, top_k)
    
    # Handle cases where no results are found
    if len(indices[0]) == 0:
        return []
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(data):
            results.append((data.iloc[idx]['content'], float(distances[0][i])))
    
    return results

# Streamlit UI
def main():
    st.title("RAG App: Chat with Your CSV")
    
    # Load models
    embedder, generator = load_models()
    
    # Session state initialization
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file:
        # Load and process the CSV
        if st.session_state.df is None:
            st.session_state.df = load_csv(uploaded_file)
            st.write("Loaded Data Preview:")
            st.dataframe(st.session_state.df.head())
            
            # Build FAISS index
            with st.spinner("Building FAISS index..."):
                st.session_state.index, _ = build_faiss_index(st.session_state.df, embedder)
            st.success("Index built successfully!")
    
        # Chat interface
        st.subheader("Chat with your CSV")
        query = st.text_input("Enter your query:")
        
        if query:
            try:
                # Retrieve relevant rows
                results = search_index(query, st.session_state.index, st.session_state.df, embedder)
                
                if not results:
                    st.warning("No relevant context found for your query. Please try again.")
                else:
                    # Generate response
                    context = " ".join([res[0] for res in results])
                    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                    
                    with st.spinner("Generating response..."):
                        response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
                    
                    # Display response
                    st.write("### Response")
                    st.write(response)
                    
                    # Display retrieved context
                    with st.expander("View Relevant Context"):
                        for res, distance in results:
                            st.write(f"- {res} (Score: {distance:.4f})")
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()