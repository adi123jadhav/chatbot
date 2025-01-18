from flask import Flask, render_template, request, jsonify,redirect
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from uuid import uuid4
from dotenv import load_dotenv
import os
import json
import requests
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
app = Flask(__name__)


# Load initial documents

pc = PINECONE_API_KEY
index = pc.Index("consumercare")
# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def is_data_already_indexed(index, expected_count):
    """Check if the dataset is already indexed."""
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) >= expected_count


loader = CSVLoader(r"FAQ_Question_Answer_Format.csv")
document = loader.load()

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

text_splitter = CharacterTextSplitter(
    separator="/n",
    chunk_size=1000,
    chunk_overlap=200
)

doc_chunks = text_splitter.split_documents(document)

# Initialize LLM and memory
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.5
)
retriever = vector_store.as_retriever()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
    memory=memory,
    verbose=True
)

if not is_data_already_indexed(index, len(doc_chunks)):
    uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    vector_store.add_documents(documents=doc_chunks, ids=uuids)
else:
    print("Data is already indexed. Skipping re-indexing.")



def format_response(response):
    """Format the LLM output for better readability."""
    # Example: Add line breaks for long responses
    response = response.replace(". ", ".\n")  # Add a new line after each sentence.

    # Example: Replace specific patterns with formatted text
    response = response.replace("•", "-")  # Change bullets to dashes if needed.

    # Example: Wrap the response in tags for HTML rendering
    response = f"<p>{response}</p>".replace("\n", "<br>")

    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat interactions."""
    user_input = request.json.get("message")
    
    # Create a prompt for the LLM
    prompt = "You are a helpful customer care executive. Act as an empathetic human and be polite and precise give structured point wise answer. User input: "
    full_input = prompt + user_input

    # Invoke the model and get a response
    output = chain.invoke({"question": full_input,"chat_history": memory.load_memory_variables({})["chat_history"]})
    raw_response = output.get("answer", "I'm sorry do you want me to connect you to an human agent?")
    
    # Format the response (example: adding line breaks or bullet points)
    formatted_response = format_response(raw_response)

    return jsonify({"response": formatted_response})




if __name__ == '__main__':
    app.run(debug=True)