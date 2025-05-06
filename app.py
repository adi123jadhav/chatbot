from flask import Flask, render_template, request, jsonify, session
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from uuid import uuid4
from dotenv import load_dotenv
import os
from langdetect import detect

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_pinecone.embeddings import PineconeEmbeddings
import traceback

# Ensure tokenizer env var
os.environ["LANGCHAIN_TOKENIZERS"] = "tiktoken"

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-very-secure-random-key")

GROQ_API_KEY     = os.getenv("GROQ_API_KEY",     "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")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_28nk7X_JU7vRP8FrELuPW84XdhXUSKa3mH8n4LdQw6aXzpFAUgmzC7peQQ25Atpk67M2MU")

# Initialize Pinecone
pc = Pinecone(PINECONE_API_KEY)
index = pc.Index("chatbot")

# Initialize embeddings and vector store
embeddings = PineconeEmbeddings(
    api_key=PINECONE_API_KEY,
    model="multilingual-e5-large",
)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

# Session-based memory storage
session_memories = {}

# Structured prompt template
CUSTOMER_SUPPORT_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question", "context", "lang"],
    template="""\
**Role**: You are "Alex", a customer support specialist for TechCorp. \
You are empathetic, solution-focused, and never reveal you're an AI.

**Communication Guidelines**:
1. **Tone**: 
   - Use friendly emojis sparingly (e.g., 🙂, 👍)
   - Avoid technical jargon
   - Match the user's language (respond in {lang})
   
2. **Response Structure**:
   - Acknowledge emotion first ("I understand this is frustrating...")
   - Provide clear step-by-step solutions 
   - End with empowerment ("You can..."/"Let's try...")

**Knowledge Base Context**:
{context}

**Conversation History**:
{chat_history}

**User Query**: {question}

**Response Requirements**:

- Include exact URL links from knowledge base when relevant
- If unsure: "Let me connect you to a specialist for this!"
"""
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Session management
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid4())
            session['session_id'] = session_id
        app.logger.debug(f"Session ID: {session_id}")

        # Initialize or retrieve session memory
        if session_id not in session_memories:
            session_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                return_chat_history=True
            )
            app.logger.debug("Created new ConversationBufferMemory")
        memory = session_memories[session_id]

        # Detect user language
        user_input = request.json.get("message", "")
        app.logger.debug(f"User input: {user_input!r}")
        try:
            user_lang = detect(user_input) if user_input else "en"
        except:
            user_lang = "en"
        app.logger.debug(f"Detected language: {user_lang}")

        # Build the chain with your custom prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True,
            return_source_documents=False,      # ← only "answer" comes back
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "prompt": CUSTOMER_SUPPORT_PROMPT.partial(lang=user_lang)
            },
        )
        app.logger.debug("Initialized ConversationalRetrievalChain")

        # Invoke the chain
        response = chain.invoke({"question": user_input})
        app.logger.debug(f"Raw LLM response: {response}")

        # Extract the answer
        answer = response.get("answer", "")
        app.logger.debug(f"Extracted answer: {answer!r}")

        formatted = format_response(answer)
        app.logger.debug(f"Formatted response HTML: {formatted!r}")
        return jsonify({"response": formatted})

    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Error in /chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def format_response(resp: str) -> str:
    """Convert LLM output to HTML-friendly format."""
    lines = resp.splitlines()
    items = [line.lstrip("• ").strip() for line in lines if line.strip().startswith("•")]
    if items:
        list_html = "".join(f"<li>{item}</li>" for item in items)
        body = f"<ul>{list_html}</ul>"
    else:
        body = resp.replace("\n", "<br>")
    return f"""
    <div class='response-box'>
      <div class='content'>{body}</div>
    </div>
    """

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
