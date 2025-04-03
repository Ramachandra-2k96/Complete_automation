import os
import re
import uuid
import csv
import time
import random
import json
import datetime
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# ANSI Color Codes for Terminal Output
COLOR_ACTOR = "\033[95m"    # Purple
COLOR_RAG = "\033[92m"      # Green
COLOR_CITATION = "\033[93m" # Yellow
COLOR_EVAL = "\033[91m"     # Red
COLOR_RESET = "\033[0m"     # Reset color

# CSV file for logging conversations
CSV_FILE = "conversation_evaluations.csv"

# -------------------------
# 1. Initialize AI Models & Embeddings
# -------------------------
# Actor LLM (plays the role of website visitor)
MODEL_NAME = "llama3.2"
EMBED_MODEL_NAME = "nomic-embed-text"

actor_llm = Ollama(model=MODEL_NAME, request_timeout=600, temperature=0.9)

# RAG LLM (answers questions about IonIdea)
rag_llm = Ollama(model="granite3.1-moe", request_timeout=600, temperature=0.3)

# Embedding model
embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

# Set default settings
Settings.embed_model = embed_model
Settings.llm = rag_llm

# -------------------------
# 2. Connect to existing ChromaDB created by new.py
# -------------------------
CHROMA_DB_PATH = "./chroma_db_IonIdea_optimized_new"  # Same path used in new.py
COLLECTION_NAME = "document_knowledge_base"

# Initialize ChromaDB client
print(f"{COLOR_CITATION}[INFO] Connecting to ChromaDB at {CHROMA_DB_PATH}{COLOR_RESET}")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"{COLOR_CITATION}[INFO] Successfully connected to collection with {collection.count()} documents{COLOR_RESET}")
except Exception as e:
    print(f"{COLOR_CITATION}[ERROR] Failed to connect to ChromaDB: {str(e)}{COLOR_RESET}")
    print(f"{COLOR_CITATION}[INFO] Please run new.py first to create the database{COLOR_RESET}")
    exit(1)

# Create vector store and prepare for index creation
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -------------------------
# 3. Create Index from Existing ChromaDB
# -------------------------
def load_index():
    """Load index from existing ChromaDB"""
    try:
        print(f"{COLOR_CITATION}[INFO] Creating index from existing ChromaDB collection...{COLOR_RESET}")
        
        # Check if the collection has any documents
        if collection.count() == 0:
            print(f"{COLOR_CITATION}[ERROR] ChromaDB collection is empty!{COLOR_RESET}")
            print(f"{COLOR_CITATION}[INFO] Please run new.py first to create and populate the database{COLOR_RESET}")
            exit(1)
            
        # Get all documents from ChromaDB
        print(f"{COLOR_CITATION}[INFO] Loading documents from ChromaDB collection...{COLOR_RESET}")
        chroma_results = collection.get(limit=collection.count())
        
        # Recreate nodes from ChromaDB data
        nodes = []
        if chroma_results and 'documents' in chroma_results and chroma_results['documents']:
            print(f"{COLOR_CITATION}[INFO] Recreating nodes from {len(chroma_results['documents'])} ChromaDB entries...{COLOR_RESET}")
            
            for i, (doc_id, doc_text, metadata) in enumerate(zip(
                chroma_results['ids'], 
                chroma_results['documents'], 
                chroma_results['metadatas']
            )):
                # Ensure all metadata values are simple types
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)) or value is None:
                        clean_metadata[key] = value
                    else:
                        # Convert complex types to string representation
                        clean_metadata[key] = str(value)
                
                # Create a new node from the document data
                node = Document(
                    text=doc_text,
                    id_=doc_id,
                    metadata=clean_metadata
                )
                nodes.append(node)
                
                if i % 100 == 0 and i > 0:
                    print(f"{COLOR_CITATION}[INFO] Recreated {i} nodes so far...{COLOR_RESET}")
            
            print(f"{COLOR_CITATION}[INFO] Successfully recreated {len(nodes)} nodes.{COLOR_RESET}")
            
            # Create a storage context with the vector store
            new_storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Add nodes to docstore
            for node in nodes:
                new_storage_context.docstore.add_documents([node])
            
            # Create index from the nodes
            index = VectorStoreIndex(
                nodes,
                storage_context=new_storage_context,
                embed_model=embed_model
            )
            
            print(f"{COLOR_CITATION}[INFO] Successfully created index with {len(new_storage_context.docstore.docs)} documents in docstore.{COLOR_RESET}")
            return index
        else:
            print(f"{COLOR_CITATION}[ERROR] No documents found in ChromaDB.{COLOR_RESET}")
            print(f"{COLOR_CITATION}[INFO] Please run new.py first to create the database properly{COLOR_RESET}")
            exit(1)
            
    except Exception as e:
        print(f"{COLOR_CITATION}[ERROR] Failed to create index: {str(e)}{COLOR_RESET}")
        print(f"{COLOR_CITATION}[INFO] Please run new.py first to create the database properly{COLOR_RESET}")
        exit(1)

# -------------------------
# 4. Setup RAG System
# -------------------------
def setup_rag_system(index):
    # System prompt for RAG model
    rag_system_prompt = """You are a concise, fact-based assistant for IonIdea.

# Guidelines for Answering Questions

- Provide specific facts, numbers, and concrete details from retrieved documents
- If no information is found, say "I don't have information about that in my documents"
- Keep responses brief and focused on answering the specific question asked
- Do not elaborate or provide general information unless specifically asked
- Structure information in readable, scannable formats (bullets/short paragraphs)
- Include precise metrics, dates, and statistics when available
- Never make up information not found in the documents
- Don't use phrases like "I understand" or "I appreciate" - just give the facts
- Avoid vague corporate language and generalizations
- Focus on specific capabilities, projects, and concrete achievements
- Provide specific details, concrete and to-the-point answers only

**Important**: Keep answers short and concise. No long summaries or detailed explanations unless explicitly requested.
"""
    # Create vector retriever with same settings as in new.py
    vector_retriever = index.as_retriever(
        similarity_top_k=6,
        embed_model=embed_model, 
        verbose=True, 
        use_recency_weighting=True,
        diversity_penalty=0.2
    )

    # For simplicity and to avoid potential BM25 errors, we'll use only the vector retriever
    # This matches one of the fallback paths in new.py
    print(f"{COLOR_CITATION}[INFO] Using vector retrieval for chat engine{COLOR_RESET}")
    retriever = vector_retriever

    # Custom prompts for better control - same as in new.py
    context_prompt = """# Context Information

---------------------
{context_str}
---------------------

## Guidelines for Response
- Answer directly and concisely based on context only
- Include specific facts, numbers and concrete details
- Structure information in readable format
- Omit unnecessary explanations and general information
"""

    context_refine_prompt = """# Question and Current Answer
- Original question: {query_str}
- Existing answer: {existing_answer}

# New Context
---------------------
{context_msg}
---------------------

## Refinement Guidelines
- Use specific facts from new context
- Include only directly relevant information
- Be concise and fact-focused
- Return original answer if context adds no value
- Keep answers short and focused
- Provide concrete, specific details
- Avoid general or vague responses
"""

    condense_prompt = """# Conversation Context
{chat_history}

# Follow-up Question
{question}

## Task
Rewrite as standalone question. Be brief, no explanations.
"""
    
    # Create a fresh memory for each conversation
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
    
    # Create chat engine with enhanced configuration - same as in new.py
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        llm=rag_llm,
        retriever=retriever,
        context_prompt=context_prompt,
        context_refine_prompt=context_refine_prompt,
        condense_prompt=condense_prompt,
        system_prompt=rag_system_prompt,
        verbose=True,  # Enable verbose output for debugging
    )
    
    return chat_engine

# -------------------------
# 5. Setup Actor System
# -------------------------
def get_actor_persona():
    personas = [
        {
            "role": "Potential Customer",
            "traits": "skeptical, detail-oriented, looking for value",
            "background": "CTO at a mid-sized financial services company looking to modernize systems",
            "communication_style": "direct, occasionally impatient, asks pointed follow-up questions",
            "description": "You're a busy CTO at a mid-sized financial services company looking to possibly hire IonIdea. You're skeptical of vendor claims and need concrete evidence of capabilities. You care about quality, timelines, and proven expertise in financial tech. Your time is valuable, so you want clear, specific answers."
        },
        {
            "role": "Job Seeker",
            "traits": "ambitious, somewhat anxious, carefully evaluating culture fit",
            "background": "Senior developer with 7 years experience, currently at a toxic workplace",
            "communication_style": "thoughtful, asks about specifics, concerned about work-life balance",
            "description": "You're a senior developer with 7 years of experience currently working at a company with a toxic environment. You're looking for a place with better work-life balance and growth opportunities. You're evaluating IonIdea carefully because you don't want to make another bad career move. You're particularly interested in company culture, technical challenges, and advancement paths."
        },
        {
            "role": "Sales Representative",
            "traits": "persistent, relationship-focused, strategic",
            "background": "Account executive at a cloud services provider looking to partner",
            "communication_style": "friendly but probing, builds rapport, focuses on pain points",
            "description": "You're an account executive at a leading cloud services provider looking to potentially partner with IonIdea or sell your services. You're skilled at building relationships and identifying business needs. Your approach is friendly but strategic - you want to understand IonIdea's challenges and decision-making process so you can position your solutions effectively."
        },
        {
            "role": "Curious Website Visitor",
            "traits": "inquisitive, somewhat tech-savvy, browsing multiple companies",
            "background": "Recent computer science graduate interested in the industry",
            "communication_style": "casual, sometimes asks basic questions, curious about details",
            "description": "You're a recent computer science graduate exploring different tech companies to learn more about the industry. You're casually browsing IonIdea's website among several others to better understand what different software companies do. You have basic technical knowledge but aren't an expert in the business side of things yet."
        }
    ]
    
    # Randomly select a persona
    persona = random.choice(personas)
    
    # Create system prompt for the actor
    actor_system_prompt = (
        f"You are a {persona['role']} visiting the IonIdea website. {persona['description']} "
        f"Your personality traits: {persona['traits']}. Your background: {persona['background']}. "
        f"Your communication style: {persona['communication_style']}. "
        "Ask natural, conversational questions as if you're typing on a website chat. "
        "Use realistic typing patterns - occasional typos, shortened words, varying punctuation, etc. "
        "Include natural conversational elements like 'hmm', 'by the way', 'actually', etc. "
        "After receiving each response, you'll internally evaluate how helpful and satisfactory it was. "
        "Rate each answer on a scale from 'perfect', 'good', 'moderate', to 'bad' based on these criteria: "
        "- PERFECT: Contains specific facts, numbers, exact details, and directly answers your question concisely. Information is structured for easy reading. No fluff or filler language. "
        "- GOOD: Provides facts and details but could be more specific or concise. Contains useful information but with some unnecessary content. "
        "- MODERATE: Has some relevant information but lacks specifics, uses vague language, or includes too much general information not directly related to your question. "
        "- BAD: Vague, uses corporate-speak, lacks concrete facts/numbers, gives generic information, or is excessively wordy with minimal actual content. "
        "Be brutally honest in your evaluation - don't be concerned about being polite. "
        "NEVER mention that you're evaluating the responses - this is strictly internal. "
        "Do not add any extra information or explanations. "
        "Do not say who you are because no humans directly tells about that and do not mention anything about evaluating the response never. "
        "IMPORTANT: Always keep your persona consistent and stay in character - you're a real person."
    )
    
    return persona, actor_system_prompt

# -------------------------
# 6. CSV Logger with Improved Citation Format
# -------------------------
def setup_csv_file():
    # Create CSV file if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Actor Role', 'Actor Background', 'Question', 'Response', 'Citations', 'Evaluation', 'Evaluation Reasoning'])
            
def log_conversation(actor_role, actor_background, question, response, citations, evaluation, evaluation_reasoning=""):
    with open(CSV_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Improve citation format - make it a JSON-formatted list for better readability
        citations_json = json.dumps(citations) if citations else "[]"
        
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            actor_role,
            actor_background,
            question,
            response,
            citations_json,
            evaluation,
            evaluation_reasoning
        ])

# -------------------------
# 7. Get Response from RAG System
# -------------------------
def get_rag_response(chat_engine, question):
    # Get response from RAG system
    response = chat_engine.chat(question)
    
    # Extract response text
    response_text = response.response
    
    # Extract citations
    citations = []
    if hasattr(response, 'source_nodes') and response.source_nodes:
        for node in response.source_nodes:
            meta = node.metadata
            file_name = meta.get("file_name", "Unknown File")
            page_label = meta.get("page_label", "Unknown Page")
            score = node.score if hasattr(node, 'score') else 0.0
            citations.append({"file": file_name, "page": page_label, "score": score})
            
    # Remove duplicates from citations while preserving scores
    unique_citations = {}
    for citation in citations:
        key = f"{citation['file']}, Page: {citation['page']}"
        if key not in unique_citations or citation['score'] > unique_citations[key]['score']:
            unique_citations[key] = citation
    
    # Convert back to list
    citations_list = list(unique_citations.values())
    
    # Sort by relevance score
    citations_list.sort(key=lambda x: x['score'] if 'score' in x else 0, reverse=True)
    
    return response_text, citations_list

# -------------------------
# 8. Get Evaluation from Actor
# -------------------------
def get_actor_evaluation(actor_system_prompt, actor_role, question, response, citations):
    # Create a prompt for the actor to evaluate the response - encouraging brutal honesty
    evaluation_prompt = (
        f"You asked the following question as a {actor_role}: '{question}'\n\n"
        f"You received this response: '{response}'\n\n"
        "Evaluate this response BRUTALLY HONESTLY. Don't worry about feelings or being polite. "
        "Judge strictly on these criteria:\n"
        "1. SPECIFICITY: Did it provide concrete facts, numbers, and specific details?\n"
        "2. CONCISENESS: Was it direct and to-the-point without unnecessary explanations?\n"
        "3. RELEVANCE: Did it directly answer your exact question without adding irrelevant information?\n"
        "4. STRUCTURE: Was information presented in an easily scannable format?\n"
        "Be EXTREMELY CRITICAL of responses that use vague corporate language, generalities, or excessive wordiness. "
        "\n\nFirst, select ONE rating from: 'perfect', 'good', 'moderate', or 'bad'."
        "\n\nThen, in a new paragraph after your rating, explain WHY you gave this rating in 1-2 sentences. "
        "Focus on specific shortcomings in factual content, conciseness, or excessive generalization."
    )
    
    # Get evaluation from actor LLM with higher temperature for more critical responses
    evaluation_response = actor_llm.complete(evaluation_prompt, temperature=0.7)
    
    # Extract just the rating (perfect, good, moderate, bad)
    rating_pattern = r'\b(perfect|good|moderate|bad)\b'
    match = re.search(rating_pattern, evaluation_response.text.lower())
    
    rating = "moderate"  # Default
    reasoning = ""
    
    if match:
        rating = match.group(0)
        
        # Try to extract reasoning if present
        parts = evaluation_response.text.split('\n\n', 1)
        if len(parts) > 1:
            reasoning = parts[1].strip()
        else:
            # If no clear paragraph break, try to get text after the rating
            rating_pos = evaluation_response.text.lower().find(rating)
            if rating_pos > -1:
                remaining = evaluation_response.text[rating_pos + len(rating):].strip()
                reasoning = remaining
    
    return rating, reasoning

# -------------------------
# 9. Main Conversation Loop
# -------------------------
def simulate_conversation(num_exchanges=5):
    # Load index from existing ChromaDB (created by new.py)
    index = load_index()
    
    # Setup CSV file
    setup_csv_file()
    
    total_conversations = num_exchanges
    
    for conversation_num in range(total_conversations):
        # Create a fresh chat engine for each conversation to prevent memory carryover
        chat_engine = setup_rag_system(index)
        
        # Get a fresh actor persona for each conversation
        persona, actor_system_prompt = get_actor_persona()
        
        print(f"\n{COLOR_CITATION}[INFO] Starting conversation {conversation_num+1}/{total_conversations}{COLOR_RESET}")
        print(f"{COLOR_CITATION}[INFO] Actor: {persona['role']} - {persona['background']}{COLOR_RESET}")
        print("-" * 80)
        
        # Prepare actor with initial prompt - more conversational
        actor_context = (
            f"{actor_system_prompt}\n\n"
            "You're now on the IonIdea website and want to learn more about the company. "
            "Ask your first question based on your role and interests. Keep it natural and conversational - "
            "as if you're typing in a live chat. Don't be overly formal; use casual language if appropriate for your persona."
        )
        
        # Get initial question from actor
        initial_response = actor_llm.complete(actor_context, temperature=0.8)
        current_question = initial_response.text
        
        # One question-answer exchange
        # Print actor's question
        print(f"{COLOR_ACTOR}[{persona['role']}]: {current_question}{COLOR_RESET}")
        
        # Get response from RAG
        rag_response, citations = get_rag_response(chat_engine, current_question)
        
        # Print RAG response
        print(f"{COLOR_RAG}[IonIdea Representative]: {rag_response}{COLOR_RESET}")
        
        # Print citations in improved format
        if citations:
            print(f"{COLOR_CITATION}Citations (by relevance):{COLOR_RESET}")
            for citation in citations:
                score = citation.get('score', 0.0)
                print(f"{COLOR_CITATION}- {citation['file']}, Page: {citation['page']} (relevance: {score:.2f}){COLOR_RESET}")
        
        # Get critical evaluation from actor
        correctness, reasoning = get_actor_evaluation(actor_system_prompt, persona['role'], current_question, rag_response, citations)
        
        # Print evaluation
        print(f"{COLOR_EVAL}[Evaluation]: {correctness}{COLOR_RESET}")
        print(f"{COLOR_EVAL}[Reasoning]: {reasoning}{COLOR_RESET}")
        
        # Log the conversation with more details
        log_conversation(
            persona['role'], 
            persona['background'], 
            current_question, 
            rag_response, 
            citations,
            correctness,
            reasoning
        )
        
        # Separation line
        print("-" * 80)
        
        # Small delay between conversations
        time.sleep(1)
        
    print(f"{COLOR_CITATION}[INFO] Completed {total_conversations} conversations. Results saved to {CSV_FILE}{COLOR_RESET}")

# -------------------------
# 10. Run the Conversation Simulation
# -------------------------
if __name__ == "__main__":
    num_exchanges = int(input(f"Enter number of conversations to simulate (default: 5): ") or "5")
    simulate_conversation(num_exchanges) 