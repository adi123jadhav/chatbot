
from flask import Flask, render_template, request, jsonify, session
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from uuid import uuid4
from dotenv import load_dotenv
import os
from langdetect import detect  # For language detection

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_pinecone.embeddings import PineconeEmbeddings
os.environ["LANGCHAIN_TOKENIZERS"] = "tiktoken"

# Load environment variables
load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = "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"
PINECONE_API_KEY = "pcsk_28nk7X_JU7vRP8FrELuPW84XdhXUSKa3mH8n4LdQw6aXzpFAUgmzC7peQQ25Atpk67M2MU"
app.secret_key = "your-very-secure-random-key"


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
- Keep under 150 words
- Use bullet points with • for complex steps
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
        
        # Initialize or retrieve session memory
        if session_id not in session_memories:
            session_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                return_chat_history=True
            )
        memory = session_memories[session_id]

        # Detect user language
        user_input = request.json.get("message")
        try:
            user_lang = detect(user_input)
        except:
            user_lang = "en"  # Fallback to English

        # Initialize chain with custom prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=False,
            return_source_documents=False,
            chain_type="stuff"  # Simpler than map_reduce
        )

        # Process query
        response = chain.invoke({
            "question": user_input,
            "lang": user_lang
        })

        # Format response
        d=response.get("answer", "I'm sorry do you want me to connect you to an human agent?")
        formatted_response = format_response(d)
        
        return jsonify({"response": formatted_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def format_response(response):
    """Convert LLM output to HTML-friendly format"""
    # Convert bullets to HTML list
    response = response.replace("• ", "<li>").replace("\n", "</li>")
    if "<li>" in response:
        response = f"<ul>{response}</ul>"
    
    # Add basic styling
    return f"""
    <div class='response-box'>
        <div class='avatar'>🤖</div>
        <div class='content'>
            {response.replace("**", "<strong>").replace("</strong>", "</strong> ")}
        </div>
    </div>
    """
if __name__ == '__main__':
    app.run(debug=True)
