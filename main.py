import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from transformers import pipeline

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "TOKEN_HERE"

# Define settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 256
Settings.chunk_overlap = 25

# Initialize the pipeline with Google Gemini model
try:
    llm_generator = pipeline(
        "text-generation",
        model="google/gemma-2b-it",
        max_length=100,
        use_auth_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
except Exception as e:
    print("Error initializing LLM:", e)

# Read and store docs into vector DB
reader = SimpleDirectoryReader(input_files=["text.txt"])
documents = reader.load_data()

# Ad hoc document refinement
for doc in documents:
    if "Member-only story" in doc.text or "The Data Entrepreneurs" in doc.text or " min read" in doc.text:
        documents.remove(doc)

index = VectorStoreIndex.from_documents(documents)

# Set up search function
top_k = 3
retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# Query documents
query = "What is iPhone 15?"
retrieved_response = query_engine.query(query)

# Combine retrieved context with the query for LLM
context = retrieved_response.get("context", "")  # Extract context if available
llm_input = f"Context: {context}\nQuestion: {query}\nAnswer:"

# Generate an answer using the LLM
try:
    response = llm_generator(llm_input, num_return_sequences=1)[0]["generated_text"]
    print("Generated Response: ", response)
except Exception as e:
    print("Error generating response:", e)
