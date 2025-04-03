import os
import uuid
from typing import List
import nltk
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.extractors.entity import EntityExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from chromadb.errors import InvalidCollectionException
import chromadb
import datetime
import hashlib
from llama_index.core import Document
from llama_parse import LlamaParse
from llama_index.llms.together import TogetherLLM

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

# ANSI Color Codes for Terminal Output
COLOR_USER = "\033[94m"     # Blue
COLOR_AI = "\033[92m"       # Green
COLOR_CITATION = "\033[93m" # Yellow
COLOR_RESET = "\033[0m"     # Reset color

# -------------------------
# 1. Initialize Model & Embeddings (Using Same Ollama Model)
# -------------------------
# Use a consistent embedding model throughout the application
MODEL_NAME = "llama3.2"
EMBED_MODEL_NAME = "nomic-embed-text"  # Keep nomic-embed-text for better embeddings

# llm = Ollama(model=MODEL_NAME, request_timeout=600)
llm = TogetherLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key="your key here"
)# Set global settings for LlamaIndex
embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

# Set as global defaults
Settings.llm = llm
Settings.embed_model = embed_model

# -------------------------
# 2. Initialize ChromaDB and Storage Context
# -------------------------
CHROMA_DB_PATH = "./chroma_db_IonIdea_optimized_new1"
COLLECTION_NAME = "document_knowledge_base"
DOCUMENTS_PATH = "Whitepaper"  # Directory containing multiple files
INDEX_PERSIST_PATH = "./storage1"

# Initialize ChromaDB client
print(f"{COLOR_CITATION}[INFO] Initializing ChromaDB at {CHROMA_DB_PATH}{COLOR_RESET}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Check if collection exists and handle properly
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"{COLOR_CITATION}[INFO] Existing collection '{COLLECTION_NAME}' found with {collection.count()} entries.{COLOR_RESET}")
    collection_exists = True
except InvalidCollectionException:
    print(f"{COLOR_CITATION}[INFO] Collection '{COLLECTION_NAME}' not found. Creating new collection.{COLOR_RESET}")
    collection = chroma_client.create_collection(COLLECTION_NAME)
    collection_exists = False
except Exception as e:
    print(f"{COLOR_CITATION}[ERROR] Unexpected error accessing ChromaDB: {str(e)}{COLOR_RESET}")
    print(f"{COLOR_CITATION}[INFO] Creating new collection as fallback.{COLOR_RESET}")
    # Attempt to create collection as fallback
    try:
        collection = chroma_client.create_collection(COLLECTION_NAME)
        collection_exists = False
    except Exception as inner_e:
        print(f"{COLOR_CITATION}[FATAL] Failed to create collection: {str(inner_e)}{COLOR_RESET}")
        raise

# Create vector store and storage context
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
parser = LlamaParse(result_type="markdown",api_key="Your key here")

# -------------------------
# 3. Initialize Chat Memory
# -------------------------
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

def find_pdf_files(directory):
    """Recursively find all PDF files in the given directory and its subdirectories."""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def get_embedding_dimension():
    """Get the embedding dimension from the model."""
    # Create a test document and get its embedding
    test_doc = Document(text="Test document")
    embedding = embed_model.get_text_embedding(test_doc.text)
    return len(embedding)
# -------------------------
# 4. Check if documents need processing
# -------------------------
def get_directory_hash(directory_path):
    """Generate a hash representing the state of all files in the directory"""
    file_hashes = []
    
    if not os.path.exists(directory_path):
        return None
        
    for root, _, files in os.walk(directory_path):
        for filename in sorted(files):  # Sort for consistency
            filepath = os.path.join(root, filename)
            # Get file modification time and size
            mtime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            # Create a hash entry for this file
            file_info = f"{filepath}:{mtime}:{size}"
            file_hashes.append(file_info)
    
    # Create a hash from all file information
    if not file_hashes:
        return None
        
    dir_info = "\n".join(file_hashes)
    return hashlib.md5(dir_info.encode()).hexdigest()

def should_process_documents(collection) -> bool:
    """
    Determine if documents need processing by checking if:
    1. Collection is empty
    2. Any document in the directory has been modified since last processing
    """
    # If collection doesn't exist or is empty, process documents
    if not collection_exists or collection.count() == 0:
        return True
    
    # Check if document directory exists
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"{COLOR_CITATION}[WARNING] Document directory '{DOCUMENTS_PATH}' not found!{COLOR_RESET}")
        return False
    
    # Get current directory hash
    current_dir_hash = get_directory_hash(DOCUMENTS_PATH)
    if not current_dir_hash:
        print(f"{COLOR_CITATION}[WARNING] Document directory is empty or could not be hashed.{COLOR_RESET}")
        return False
    
    # Query collection for metadata about last processing
    try:
        # Get a sample entry to check metadata
        results = collection.get(limit=1)
        if results and 'metadatas' in results and results['metadatas']:
            last_dir_hash = results['metadatas'][0].get('directory_hash', '')
            # If directory has changed since last processing, reprocess
            if last_dir_hash != current_dir_hash:
                print(f"{COLOR_CITATION}[INFO] Document directory has changed since last processing.{COLOR_RESET}")
                return True
        else:
            # No metadata found, process to be safe
            return True
    except Exception as e:
        print(f"{COLOR_CITATION}[WARNING] Error checking document modification time: {str(e)}{COLOR_RESET}")
        # On error, choose to process documents to be safe
        return True
        
    return False

# -------------------------
# 5. Document Processing and Indexing
# -------------------------
def process_documents() -> VectorStoreIndex:
    """Process documents and create a vector store index"""
    # Ensure directory exists
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"{COLOR_CITATION}[ERROR] Document directory '{DOCUMENTS_PATH}' not found!{COLOR_RESET}")
        print(f"{COLOR_CITATION}[INFO] Please check the path and try again.{COLOR_RESET}")
        raise FileNotFoundError(f"Directory not found: {DOCUMENTS_PATH}")
        
    # Download NLTK resources if needed
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"{COLOR_CITATION}[WARNING] Failed to download NLTK resources: {str(e)}{COLOR_RESET}")
        print(f"{COLOR_CITATION}[INFO] Continuing with processing...{COLOR_RESET}")
    
    # Get directory hash for metadata
    current_dir_hash = get_directory_hash(DOCUMENTS_PATH)
    
    # Load documents from the specified directory
    try:
        # Load all documents in directory - use correct path
        pdf_files = find_pdf_files(DOCUMENTS_PATH)
        documents = []
        
        for pdf_file in pdf_files:
            try:
                # Parse PDF using LlamaParse
                parsed_doc = parser.load_data(pdf_file)
                # Add file path to metadata
                for idx, doc in enumerate(parsed_doc):
                    # Set consistent metadata
                    doc.metadata["file_path"] = pdf_file
                    doc.metadata["file_name"] = os.path.basename(pdf_file)
                    doc.metadata['directory_hash'] = current_dir_hash
                    doc.metadata['processed_date'] = datetime.datetime.now().isoformat()
                    
                    # LlamaParse typically includes page information in metadata 
                    if "page_label" not in doc.metadata:
                        # First check if it's directly available
                        if hasattr(doc, "page_label"):
                            doc.metadata["page_label"] = str(doc.page_label)
                        # Check page_number in metadata
                        elif "page_number" in doc.metadata:
                            doc.metadata["page_label"] = str(doc.metadata["page_number"])
                        # Try to get from the document's page_number attribute
                        elif hasattr(doc, "page_number"):
                            doc.metadata["page_label"] = str(doc.page_number)
                        # If page info is in metadata under a different key
                        elif any(key for key in doc.metadata if "page" in key.lower()):
                            for key in doc.metadata:
                                if "page" in key.lower():
                                    doc.metadata["page_label"] = str(doc.metadata[key])
                                    break
                        # Last resort: use document index as a fallback
                        else:
                            doc.metadata["page_label"] = str(idx + 1)
                    
                    # Ensure page_label is a string
                    if "page_label" in doc.metadata and not isinstance(doc.metadata["page_label"], str):
                        doc.metadata["page_label"] = str(doc.metadata["page_label"])
                        
                    # Add file modification time if file exists
                    if os.path.exists(pdf_file):
                        doc.metadata['last_modified'] = str(os.path.getmtime(pdf_file))
                        
                    # Print debugging info to verify page labels are extracted
                    print(f"{COLOR_CITATION}[DEBUG] Document {idx+1} page_label: {doc.metadata.get('page_label', 'Not found')}{COLOR_RESET}")
                    
                documents.extend(parsed_doc)
                print(f"{COLOR_CITATION}[INFO] Successfully parsed: {pdf_file}{COLOR_RESET}")
            except Exception as e:
                print(f"{COLOR_CITATION}[ERROR] Failed to parse {pdf_file}: {str(e)}{COLOR_RESET}")

        print(f"{COLOR_CITATION}[INFO] Loaded {len(documents)} documents.{COLOR_RESET}")
        
        if not documents:
            print(f"{COLOR_CITATION}[WARNING] No documents loaded from '{DOCUMENTS_PATH}'{COLOR_RESET}")
            raise ValueError("No documents found to process")
        
        # The rest of the document metadata processing is redundant - we already set it above
        # We'll skip this section and move directly to node creation
    except Exception as e:
        print(f"{COLOR_CITATION}[ERROR] Failed to load documents: {str(e)}{COLOR_RESET}")
        raise
    
    # Define a more effective node parser for better chunking
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        paragraph_separator="\n\n",
        separator=" "
    )
    
    # Define extractors for rich metadata extraction
    extractors = [
        node_parser,
        QuestionsAnsweredExtractor(questions=8),
        TitleExtractor(nodes=5),
        SummaryExtractor(summaries=["self"]),
        KeywordExtractor(keywords=8),
    ]
    
    # Create ingestion pipeline with vector store
    pipeline = IngestionPipeline(
        transformations=extractors,
        vector_store=vector_store
    )
    
    # Clear existing collection if needed
    if collection.count() > 0:
        print(f"{COLOR_CITATION}[INFO] Clearing existing collection entries...{COLOR_RESET}")
        collection.delete(where={})
    
    # Process documents into nodes with rich metadata
    print(f"{COLOR_CITATION}[INFO] Processing documents into nodes...{COLOR_RESET}")
    try:
        nodes = pipeline.run(documents=documents)
        print(f"{COLOR_CITATION}[INFO] Created {len(nodes)} nodes from documents.{COLOR_RESET}")
        
        if not nodes:
            print(f"{COLOR_CITATION}[WARNING] No nodes were created from the documents.{COLOR_RESET}")
            raise ValueError("Document processing yielded no nodes")
            
        # Store document embeddings in ChromaDB
        batch_size = 100  # Process in batches to avoid memory issues
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            
            ids = []
            documents_text = []
            embeddings = []
            metadatas = []
            
            for node in batch:
                metadata = node.metadata
                filename = metadata.get("file_path", "Unknown File")
                page = metadata.get("page_label", "Unknown Page")
                content = node.text
                doc_id = str(uuid.uuid4())  # Unique ID

                # Generate embedding for the node
                embedding = embed_model.get_text_embedding(content)
                
                # Set the node ID in the node itself (important for docstore)
                node.id_ = doc_id
                
                # Add to batch
                ids.append(doc_id)
                documents_text.append(content)
                embeddings.append(embedding)
                
                # Clean metadata to ensure only simple types are stored
                clean_metadata = {
                    "file_name": os.path.basename(filename),
                    "file_path": filename,
                    "page_label": str(page),
                    "directory_hash": current_dir_hash,
                    "node_id": doc_id,
                    "text_length": len(content),
                    "processed_date": datetime.datetime.now().isoformat()
                }
                
                metadatas.append(clean_metadata)
            
            # Store batch in ChromaDB
            collection.add(
                ids=ids,
                documents=documents_text,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"{COLOR_CITATION}[INFO] Stored batch of {len(batch)} nodes with embeddings{COLOR_RESET}")
            
    except Exception as e:
        print(f"{COLOR_CITATION}[ERROR] Failed during document processing: {str(e)}{COLOR_RESET}")
        raise
    
    # Create index from vector store and nodes
    try:
        print(f"{COLOR_CITATION}[INFO] Creating vector index...{COLOR_RESET}")
        
        # Create a storage context that includes the nodes
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Add nodes to docstore
        for node in nodes:
            storage_context.docstore.add_documents([node])
        
        # Create index with the storage context that has nodes in the docstore
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Ensure directory exists for persistence
        os.makedirs(INDEX_PERSIST_PATH, exist_ok=True)
        
        # Persist the index
        index.storage_context.persist(persist_dir=INDEX_PERSIST_PATH)
        print(f"{COLOR_CITATION}[INFO] Index created with {len(nodes)} nodes and persisted to {INDEX_PERSIST_PATH}{COLOR_RESET}")
        
        return index
    except Exception as e:
        print(f"{COLOR_CITATION}[ERROR] Failed to create index: {str(e)}{COLOR_RESET}")
        raise

# -------------------------
# 6. Load or Create Index
# -------------------------
try:
    if should_process_documents(collection):
        print(f"{COLOR_CITATION}[INFO] Documents need processing. Creating new index...{COLOR_RESET}")
        index = process_documents()
    else:
        print(f"{COLOR_CITATION}[INFO] Using existing index from ChromaDB...{COLOR_RESET}")
        # Try to load from persisted storage first
        if os.path.exists(INDEX_PERSIST_PATH):
            print(f"{COLOR_CITATION}[INFO] Loading persisted index from {INDEX_PERSIST_PATH}...{COLOR_RESET}")
            
            # Get all documents from ChromaDB
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
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=INDEX_PERSIST_PATH
                )
                
                # Add nodes to docstore
                for node in nodes:
                    storage_context.docstore.add_documents([node])
                
                # Create index from the nodes
                index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
                
                print(f"{COLOR_CITATION}[INFO] Loaded index with {len(storage_context.docstore.docs)} documents in docstore.{COLOR_RESET}")
            else:
                print(f"{COLOR_CITATION}[WARNING] No documents found in ChromaDB. Creating empty index...{COLOR_RESET}")
                # Create empty index with storage context
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=INDEX_PERSIST_PATH
                )
                index = VectorStoreIndex([], storage_context=storage_context)
        else:
            # Fallback to loading directly from vector store (this path may not populate the docstore)
            print(f"{COLOR_CITATION}[WARNING] No persisted index found. Loading directly from ChromaDB...{COLOR_RESET}")
            
            # Get all documents from ChromaDB
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
                
                # Create storage context
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Add nodes to docstore
                for node in nodes:
                    storage_context.docstore.add_documents([node])
                
                # Create index from the nodes
                index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
                
                print(f"{COLOR_CITATION}[INFO] Created index with {len(storage_context.docstore.docs)} documents in docstore.{COLOR_RESET}")
            else:
                print(f"{COLOR_CITATION}[WARNING] No documents found in ChromaDB. Creating empty index...{COLOR_RESET}")
                index = VectorStoreIndex([], storage_context=StorageContext.from_defaults(vector_store=vector_store))
except Exception as e:
    print(f"{COLOR_CITATION}[FATAL] Failed to initialize index: {str(e)}{COLOR_RESET}")
    print(f"{COLOR_CITATION}[INFO] Exiting program...{COLOR_RESET}")
    exit(1)

# -------------------------
# 7. Configure Chat Engine
# -------------------------
system_prompt = """You are an AI assistant specializing in document retrieval and question answering.
When answering questions:
1. Use retrieved document information when relevant.
2. If no information is found or the question isn't about the documents, simply say "I don't have information about that in my documents."
3. Determine if the query requires document knowledge or is general conversation.
4. Do not include page references directly in your responses.
5. Provide clear, concise answers based on the document content.
6. For general conversation, respond naturally and politely.
7. Never make up information that is not in the documents.
"""
# BM25 for keyword search
bm25_retriever = BM25Retriever.from_defaults(index)

# Create vector retriever
vector_retriever = index.as_retriever(
    similarity_top_k=6,
    embed_model=embed_model, 
    verbose=True, 
    use_recency_weighting=True,
    diversity_penalty=0.2
)

# Combine retrievers using QueryFusionRetriever for hybrid search
retriever = QueryFusionRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    similarity_top_k=8,
    num_queries=2,
    mode="simple"  # Use simple mode instead of fusion
)

# Custom prompts for better control
context_prompt = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the question.
"""

context_refine_prompt = """The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
---------------------
{context_msg}
---------------------
Given the new context, refine the original answer to better answer the question.
If the context isn't useful, return the original answer.
"""

condense_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,No futher context is needed do not explain anything just the question or the refrased prompt.
Chat History:
{chat_history}
Follow Up Question: {question}
Standalone question:
"""

# Create chat engine with enhanced configuration
chat_engine = CondensePlusContextChatEngine.from_defaults(
    llm=llm,
    retriever=retriever,
    memory=chat_memory,
    context_prompt=context_prompt,
    context_refine_prompt=context_refine_prompt,
    condense_prompt=condense_prompt,
    system_prompt=system_prompt,
    verbose=True,  # Enable verbose output for debugging
)

# -------------------------
# 8. Interactive Chat Function
# -------------------------
def chat_with_documents(query: str) -> None:
    """Process a user query and display the response with citations"""
    try:
        # Get streaming response from chat engine
        response = chat_engine.stream_chat(query)
        
        # Print AI response
        print(f"{COLOR_AI}AI: {COLOR_RESET}", end="", flush=True)
        
        # Handle streaming response
        response_text = ""
        for chunk in response.response_gen:
            response_text += chunk
            print(chunk, end="", flush=True)
        print()  # New line after response
        
        # Only show citations if we have a non-empty response
        if response_text.strip() and hasattr(response, 'source_nodes') and response.source_nodes:
            unique_citations = {}
            for node in response.source_nodes:
                meta = node.metadata
                file_name = meta.get("file_name", "Unknown File")
                page_label = meta.get("page_label", "Unknown Page")
                citation_key = f"{file_name}, Page: {page_label}"
                # Store with relevance score
                if citation_key not in unique_citations:
                    unique_citations[citation_key] = node.score if hasattr(node, 'score') else 0.0
            
            # Print citations sorted by relevance score
            if unique_citations:
                print(f"\n{COLOR_CITATION}Citations (by relevance):{COLOR_RESET}")
                for citation, score in sorted(unique_citations.items(), key=lambda x: x[1], reverse=True):
                    print(f"{COLOR_CITATION}- {citation} (relevance: {score:.2f}){COLOR_RESET}")
        
        print("\n" + "-" * 50)
    except Exception as e:
        print(f"{COLOR_CITATION}Error processing query: {str(e)}{COLOR_RESET}")
        print(f"{COLOR_AI}AI: I'm sorry, I encountered an error while processing your question. Please try again or rephrase your query.{COLOR_RESET}")
        print("\n" + "-" * 50)

# -------------------------
# 9. Main Chat Loop
# -------------------------
if __name__ == "__main__":
    print(f"{COLOR_CITATION}Optimized Document Q&A System Initialized.{COLOR_RESET}")
    print(f"{COLOR_CITATION}Using LLM: {MODEL_NAME}, Embedding: {EMBED_MODEL_NAME}{COLOR_RESET}")
    print(f"{COLOR_CITATION}Document directory path: {DOCUMENTS_PATH}{COLOR_RESET}")
    print(f"{COLOR_CITATION}Type 'exit' to quit.{COLOR_RESET}")
    
    while True:
        try:
            user_query = input(f"\n{COLOR_USER}Your question: {COLOR_RESET}")
            if user_query.lower() in ["exit", "quit", "bye"]:
                print(f"{COLOR_CITATION}Goodbye!{COLOR_RESET}")
                break
            chat_with_documents(user_query)
        except KeyboardInterrupt:
            print(f"\n{COLOR_CITATION}Interrupted by user. Exiting...{COLOR_RESET}")
            break
        except Exception as e:
            print(f"{COLOR_CITATION}Unexpected error: {str(e)}{COLOR_RESET}")
            print(f"{COLOR_CITATION}Continuing...{COLOR_RESET}")