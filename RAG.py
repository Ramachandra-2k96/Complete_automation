from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain_community.utilities.nasa import NasaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
try:
    # Use newer imports if available
    from langchain_chroma import Chroma
except ImportError:
    # Fall back to legacy import
    from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
try:
    # Use newer imports if available
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
except ImportError:
    # Fall back to legacy import
    from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent, Tool, AgentType
from langchain.agents import initialize_agent
from langchain.agents import tool as tool_decorator
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.base import ToolException
import os
import json
import uuid
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import time  # Add this import at the top if not already present

# Ensure psutil is installed for memory monitoring
try:
    import psutil
except ImportError:
    print("Installing psutil for memory monitoring...")
    import subprocess
    subprocess.check_call(["pip", "install", "psutil"])
    import psutil

import gc

# Initialize the RedditSearchRun tool
try:
    redit_search_tool = RedditSearchRun(
        name="reddit_search",
        description="Search Reddit posts by query, time, subreddit, and sort order.",
        api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id="8V_NiMJIavgaQOGxB0wb4A",
            reddit_client_secret="UR70jVsAg9YaqHmrAxgrLDouKsZBig",
            reddit_user_agent="arhgejthej",
        )
    )
except Exception as e:
    print(f"Error initializing Reddit search: {str(e)}")
    redit_search_tool = None

# Initialize NASA tools
try:
    nasa = NasaAPIWrapper()
    nasa_toolkit = NasaToolkit.from_nasa_api_wrapper(nasa)
except Exception as e:
    print(f"Error initializing NASA tools: {str(e)}")
    nasa_toolkit = None

# Initialize StackExchange
try:
    stackexchange = StackExchangeAPIWrapper(max_results=7)
except Exception as e:
    print(f"Error initializing StackExchange: {str(e)}")
    stackexchange = None

# Initialize Wikipedia and Wikidata
try:
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=8, doc_content_chars_max=400000))
except Exception as e:
    print(f"Error initializing Wikipedia: {str(e)}")
    wikipedia = None

try:
    wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper(top_k_results=8, doc_content_chars_max=400000))
except Exception as e:
    print(f"Error initializing Wikidata: {str(e)}")
    wikidata = None

# Initialize Yahoo Finance
try:
    tools_finance = YahooFinanceNewsTool()
except Exception as e:
    print(f"Error initializing Yahoo Finance: {str(e)}")
    tools_finance = None

# Initialize Arxiv
try:
    tool_arxive = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=8, max_results=100, sort_by="lastUpdatedDate", doc_content_chars_max=90000))
except Exception as e:
    print(f"Error initializing Arxiv: {str(e)}")
    tool_arxive = None

# Initialize DuckDuckGo
try:
    wrapper = DuckDuckGoSearchAPIWrapper(time="d", max_results=5)
    web_search = DuckDuckGoSearchResults(api_wrapper=wrapper)
except Exception as e:
    print(f"Error initializing DuckDuckGo: {str(e)}")
    web_search = None

# Create a dummy search tool as fallback
@tool_decorator("Search the web")
def web_search_fallback(query: str) -> str:
    """Search the web for information about the query."""
    return f"Web search results for: {query}\n\nThis is a fallback search tool that simulates web search results."

# Collect working tools
research_tools = []
for tool in [tool_arxive, wikipedia, wikidata, tools_finance, redit_search_tool, stackexchange, nasa_toolkit, web_search]:
    if tool is not None:
        research_tools.append(tool)

# Ensure we have at least the fallback
if not research_tools:
    print("Warning: No tools initialized successfully. Using fallback web search tool.")
    research_tools = [web_search_fallback]
else:
    print(f"Successfully initialized {len(research_tools)} research tools")

from langchain_groq import ChatGroq
llm = ChatGroq(api_key="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", model="llama3-8b-8192")

# Setup vector database - Make embeddings optional
USE_EMBEDDINGS = False  # Set to False for much faster processing without embeddings
SEARCH_QUERIES_LIMIT = 10  # Reduce from 90 to 10 (adjust as needed for speed vs. completeness)

if USE_EMBEDDINGS:
    try:
        # Only create embeddings if enabled
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # Force CPU usage
            encode_kwargs={"batch_size": 8}  # Smaller batch size for CPU
        )
        
        # Create a directory for persisting the vector database
        PERSIST_DIRECTORY = "research_db"
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize the vector database with CPU-friendly settings
        try:
            vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY, 
                embedding_function=embedding_function,
                collection_metadata={"hnsw:space": "cosine"}  # CPU-friendly distance metric
            )
            print(f"Vector database initialized at {PERSIST_DIRECTORY} (CPU mode)")
        except Exception as e:
            print(f"Error initializing vector database: {str(e)}")
            print("Falling back to in-memory vector store (CPU mode)")
            vectordb = Chroma(
                embedding_function=embedding_function,
                collection_metadata={"hnsw:space": "cosine"}  # CPU-friendly distance metric
            )
    except Exception as e:
        print(f"Error setting up embeddings: {str(e)}")
        USE_EMBEDDINGS = False  # Disable if setup failed
        vectordb = None
else:
    print("Embeddings disabled for faster processing - using simple file-based storage")
    vectordb = None

# Text splitter for chunking documents - optimized for CPU
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks for CPU
    chunk_overlap=50,  # Reduced overlap for CPU
    separators=["\n\n", "\n", ". ", " ", ""],
)

class ResearchWorkflow:
    def __init__(self, topic, llm=llm, tools=research_tools, max_iterations=30, db=vectordb):
        self.topic = topic
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.db = db
        self.session_id = str(uuid.uuid4())
        self.research_log = []
        
        # Configuration settings
        self.use_embeddings = USE_EMBEDDINGS
        self.search_limit = SEARCH_QUERIES_LIMIT
        self.cpu_only = True
        self.batch_size = 2  # Very small batches for CPU-only
        
        # Setup the research directory
        self.research_dir = f"research_sessions/{self.session_id}"
        os.makedirs(self.research_dir, exist_ok=True)
        os.makedirs(f"{self.research_dir}/results", exist_ok=True)  # Directory for tool results
        
        # Convert multi-input tools to single-input
        self.prepare_tools()
        
    def prepare_tools(self):
        """Convert multi-input tools to single-input format compatible with ZeroShotAgent"""
        compatible_tools = []
        
        for tool in self.tools:
            try:
                if tool.name == "reddit_search":
                    # Create a simplified wrapper for reddit search
                    def create_reddit_wrapper(reddit_tool):
                        @tool_decorator("Search Reddit for information")
                        def search_reddit(query: str) -> str:
                            """Search Reddit for the given query and return relevant posts."""
                            try:
                                return reddit_tool.run({"query": query, "sort": "relevance", "time_filter": "month"})
                            except Exception as e:
                                return f"Error searching Reddit: {str(e)}"
                        return search_reddit
                    
                    compatible_tools.append(create_reddit_wrapper(tool))
                elif hasattr(tool, "run") and callable(tool.run):
                    # Create a simplified version of the tool with a single input
                    def create_simplified_wrapper(t):
                        tool_name = getattr(t, "name", t.__class__.__name__)
                        tool_description = getattr(t, "description", f"Tool for {tool_name}")
                        
                        @tool_decorator(tool_description)
                        def simplified_tool(query: str) -> str:
                            """Run tool with a single query parameter."""
                            try:
                                # First try running with the query directly
                                return t.run(query)
                            except Exception as e1:
                                try:
                                    # Then try with the query as a parameter dict
                                    return t.run({"query": query})
                                except Exception as e2:
                                    return f"Error using {tool_name}: {str(e2)}"
                        
                        # Set the name to match the original tool
                        simplified_tool.name = tool_name
                        return simplified_tool
                    
                    compatible_tools.append(create_simplified_wrapper(tool))
                else:
                    # Tool doesn't have a run method, skip it
                    print(f"Skipping tool {getattr(tool, 'name', tool.__class__.__name__)} - no run method")
            except Exception as e:
                print(f"Error processing tool: {str(e)}")
                # Skip this tool
        
        if not compatible_tools:
            print("Warning: No compatible tools found. Using a simple web search tool as fallback.")
            @tool_decorator("Search the web")
            def web_search_fallback(query: str) -> str:
                """Search the web for information about the query."""
                try:
                    return f"Web search results for: {query}\n\nPlease check online sources for this information."
                except Exception as e:
                    return f"Error searching the web: {str(e)}"
            
            compatible_tools.append(web_search_fallback)
        
        self.tools = compatible_tools
        print(f"Prepared {len(self.tools)} tools for research")
        
    def save_to_vector_db(self, content, metadata=None):
        """Save content to vector database with CPU optimizations"""
        if not self.use_embeddings or self.db is None:
            # Save to file instead when embeddings are disabled
            if not metadata:
                metadata = {"source": "research", "timestamp": datetime.now().isoformat()}
            
            # Create a filename from question/query
            filename = f"{self.research_dir}/results/{metadata.get('tool', 'unknown')}_"
            if 'query' in metadata:
                # Sanitize query for filename
                query = metadata['query'].replace('/', '_').replace('\\', '_')[:50]
                filename += f"{query}"
            filename += f"_{uuid.uuid4().hex[:8]}.txt"
            
            # Save content to file
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"METADATA: {json.dumps(metadata)}\n\n")
                    f.write(content)
                return 1
            except Exception as e:
                print(f"Error saving to file: {str(e)}")
                return 0
        
        # If embeddings are enabled, proceed with vector DB
        if not content or not isinstance(content, str) or content.strip() == "":
            print("Warning: Empty content received, skipping vector DB storage")
            return 0
            
        if not metadata:
            metadata = {"source": "research", "timestamp": datetime.now().isoformat()}
        
        try:
            # Split text into smaller chunks for CPU processing
            docs = text_splitter.create_documents([content], [metadata])
            
            # Add documents to vector store in smaller batches to reduce memory usage
            if len(docs) > self.batch_size:
                print(f"Processing {len(docs)} chunks in batches of {self.batch_size} for CPU efficiency")
                for i in range(0, len(docs), self.batch_size):
                    batch = docs[i:i+self.batch_size]
                    self.db.add_documents(batch)
                    # Force garbage collection to reduce memory pressure
                    import gc
                    gc.collect()
            else:
                self.db.add_documents(docs)
            
            self.db.persist()
            
            return len(docs)
        except Exception as e:
            print(f"Error saving to vector database: {str(e)}")
            return 0
    
    def plan_research(self):
        """Create a research plan based on the topic"""
        plan_prompt = ChatPromptTemplate.from_template("""
        You are a world-class researcher planning a focused research project on: {topic}
        
        Create a concise research plan including:
        1. 5-7 specific research questions we should answer
        2. For each question, list 1-2 specific search queries we should run
        3. Specific data sources or tools that would be most valuable for each question (arxiv, wikipedia, web search, etc.)
        
        Your plan should be designed to gather approximately {search_limit} search results across different sources.
        
        Make your research questions focused on the most important aspects of the topic.
        
        Format your response as JSON with the following structure:
        {{
            "research_questions": [
                {{
                    "question": "The specific research question",
                    "search_queries": ["query1", "query2"],
                    "data_sources": ["tool1", "tool2"]
                }}
            ]
        }}
        """)
        
        try:
            chain = plan_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"topic": self.topic, "search_limit": self.search_limit})
            
            # Parse the JSON result
            try:
                # First try parsing as JSON
                plan = json.loads(result)
                # Save the plan
                with open(f"{self.research_dir}/research_plan.json", "w") as f:
                    json.dump(plan, f, indent=2)
                return plan
            except json.JSONDecodeError as json_error:
                print(f"Failed to parse research plan as JSON: {str(json_error)}")
                
                # Try to extract JSON portion if it exists
                try:
                    # Sometimes the model returns markdown-wrapped JSON, try to extract it
                    import re
                    json_pattern = r'```(?:json)?\s*(.*?)```'
                    matches = re.findall(json_pattern, result, re.DOTALL)
                    
                    if matches:
                        # Try each potential JSON match
                        for match in matches:
                            try:
                                plan = json.loads(match.strip())
                                # Save the plan if successful
                                with open(f"{self.research_dir}/research_plan.json", "w") as f:
                                    json.dump(plan, f, indent=2)
                                return plan
                            except:
                                continue
                
                    # If we couldn't extract JSON, create a simple fallback plan
                    fallback_plan = self._create_fallback_plan()
                    with open(f"{self.research_dir}/research_plan_fallback.json", "w") as f:
                        json.dump(fallback_plan, f, indent=2)
                    
                    # Still save the original response
                    with open(f"{self.research_dir}/research_plan_raw.txt", "w") as f:
                        f.write(result)
                    
                    return fallback_plan
                except Exception as extract_error:
                    print(f"Error extracting JSON: {str(extract_error)}")
                    
                    # Create fallback plan
                    fallback_plan = self._create_fallback_plan()
                    with open(f"{self.research_dir}/research_plan_fallback.json", "w") as f:
                        json.dump(fallback_plan, f, indent=2)
                    
                    # Save the raw response
                    with open(f"{self.research_dir}/research_plan_raw.txt", "w") as f:
                        f.write(result)
                    
                    return fallback_plan
        except Exception as e:
            print(f"Error generating research plan: {str(e)}")
            fallback_plan = self._create_fallback_plan()
            return fallback_plan
    
    def _create_fallback_plan(self):
        """Create a simple fallback research plan when the LLM fails"""
        topic_words = self.topic.split()
        
        # Create some basic research questions based on the topic
        questions = [
            f"What is the current state of {self.topic}?",
            f"What are the key challenges in {self.topic}?",
            f"How has {self.topic} evolved over time?",
            f"What are the future trends in {self.topic}?",
            f"Who are the key experts or organizations in {self.topic}?",
        ]
        
        # Add some topic-specific questions if possible
        if len(topic_words) >= 2:
            questions.append(f"How does {topic_words[0]} relate to {topic_words[-1]}?")
            questions.append(f"What are the ethical considerations in {self.topic}?")
            questions.append(f"What methodologies are used in {self.topic} research?")
        
        # Create a simple plan with these questions
        fallback_plan = {
            "research_questions": []
        }
        
        for q in questions:
            # Create search queries from the question
            search_queries = [
                q,  # The question itself
                f"latest research {q.lower().replace('?', '')}",
                f"examples of {q.lower().replace('?', '')}",
                f"{self.topic} case studies",
            ]
            
            fallback_plan["research_questions"].append({
                "question": q,
                "search_queries": search_queries,
                "data_sources": ["web_search", "wikipedia", "arxiv"]
            })
        
        return fallback_plan
    
    def execute_research(self, research_plan):
        """Execute the research plan and store results in vector database"""
        results_count = 0
        research_results = []
        
        print(f"Starting focused research with target of {self.search_limit} searches...")
        
        # Use direct tool research approach, using specific tools per query
        try:
            # Limit the plan to the configured number of searches
            total_queries = 0
            limited_plan = {"research_questions": []}
            
            for question_obj in research_plan.get("research_questions", []):
                question = question_obj.get("question", "")
                search_queries = question_obj.get("search_queries", [])
                data_sources = question_obj.get("data_sources", [])
                
                # Limit queries per question
                if total_queries + len(search_queries) <= self.search_limit:
                    # Add all queries for this question
                    limited_plan["research_questions"].append(question_obj)
                    total_queries += len(search_queries)
                else:
                    # Add only some queries to stay within limit
                    remaining = self.search_limit - total_queries
                    if remaining > 0:
                        question_obj["search_queries"] = search_queries[:remaining]
                        limited_plan["research_questions"].append(question_obj)
                        total_queries += remaining
                    break
            
            print(f"Limited research plan to {total_queries} queries to stay within search limit of {self.search_limit}")
            
            # Execute the limited research plan
            for question_obj in tqdm(limited_plan.get("research_questions", []), desc="Researching questions"):
                question = question_obj.get("question", "")
                search_queries = question_obj.get("search_queries", [])
                data_sources = question_obj.get("data_sources", [])
                
                question_results = []
                
                for query in tqdm(search_queries, desc=f"Query: {question[:30]}..."):
                    # Execute direct tool research with specified data sources
                    direct_result = self._direct_tool_research(question, query, data_sources)
                    question_results.append(direct_result)
                    results_count += 1
                
                research_results.append({
                    "question": question,
                    "results": question_results
                })
                
                # Save incremental results
                with open(f"{self.research_dir}/research_results.json", "w") as f:
                    json.dump(research_results, f, indent=2)
                
                # Report progress
                print(f"Completed research on question: {question}")
                print(f"Total searches so far: {results_count}")
        
        except Exception as e:
            print(f"Error during research execution: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Log total research statistics
        print(f"Research complete. Total searches performed: {results_count}")
        research_stats = {
            "total_searches": results_count,
            "completion_time": datetime.now().isoformat(),
            "research_questions_count": len(research_plan.get("research_questions", [])),
            "session_id": self.session_id
        }
        
        with open(f"{self.research_dir}/research_stats.json", "w") as f:
            json.dump(research_stats, f, indent=2)
        
        self.research_log = research_results
        return {"total_results": results_count, "research_log": research_results}
    
    def _direct_tool_research(self, question, query, data_sources=None):
        """Use specific tools requested for the question, not all tools"""
        combined_results = ""
        
        # Monitor memory at start
        self.print_memory_usage()
        
        # Determine which tools to use for this query
        tools_to_use = []
        if data_sources and isinstance(data_sources, list) and data_sources:
            # Only use specified tools
            for tool in self.tools:
                for source in data_sources:
                    source_lower = source.lower()
                    tool_name_lower = tool.name.lower()
                    
                    # Match tool names to requested sources
                    if (source_lower in tool_name_lower or 
                        (source_lower == "wikipedia" and "wikipedia" in tool_name_lower) or
                        (source_lower == "arxiv" and "arxiv" in tool_name_lower) or
                        (source_lower == "web" and ("web" in tool_name_lower or "search" in tool_name_lower)) or
                        (source_lower == "reddit" and "reddit" in tool_name_lower) or
                        (source_lower == "finance" and "finance" in tool_name_lower) or
                        (source_lower == "stackexchange" and "stackexchange" in tool_name_lower)):
                        tools_to_use.append(tool)
                        break
        
        # If no specific tools matched, use a default web search
        if not tools_to_use:
            for tool in self.tools:
                if "web" in tool.name.lower() or "search" in tool.name.lower():
                    tools_to_use.append(tool)
                    break
            
            # If still no tools, just use the first available
            if not tools_to_use and self.tools:
                tools_to_use = [self.tools[0]]
        
        print(f"Using {len(tools_to_use)}/{len(self.tools)} tools for query: {query}")
        
        try:
            # Try each selected tool and save results independently
            for tool_idx, tool in enumerate(tools_to_use):
                try:
                    print(f"Trying tool {tool_idx+1}/{len(tools_to_use)}: {tool.name} for query: {query}")
                    result = tool.run(query)
                    
                    # Store each tool result separately
                    tool_result = f"Research Question: {question}\nQuery: {query}\nTool: {tool.name}\n\nRESULTS:\n{result}"
                    
                    # Save with tool-specific metadata
                    metadata = {
                        "question": question,
                        "query": query,
                        "tool": tool.name,
                        "timestamp": datetime.now().isoformat(),
                        "session_id": self.session_id
                    }
                    self.save_to_vector_db(tool_result, metadata)
                    
                    # Add to combined results for logging
                    combined_results += f"\n\n--- {tool.name} RESULTS SAVED ---\n"
                    
                    # Force cleanup after each tool use
                    if tool_idx % 3 == 0:  # Every 3 tools
                        gc.collect()
                        
                except Exception as tool_error:
                    print(f"Error using tool {tool.name}: {str(tool_error)}")
            
            # Final cleanup
            gc.collect()
            self.print_memory_usage()
            
            # Create result entry for logging only
            return {
                "question": question,
                "query": query,
                "result_summary": f"Searched with {len(tools_to_use)} tools, results stored",
                "timestamp": datetime.now().isoformat(),
                "method": "direct_tool"
            }
        except Exception as e:
            print(f"Error in direct tool research: {str(e)}")
            return {
                "question": question,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "method": "direct_tool_error"
            }
    
    def plan_report(self):
        """Plan the structure of the final report"""
        if not self.use_embeddings:
            # Without embeddings, use a simpler approach to plan the report
            print("Planning report without embeddings...")
            
            # Collect all research results from files
            all_files = []
            results_dir = f"{self.research_dir}/results"
            if os.path.exists(results_dir):
                all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.txt')]
            
            # Gather information about the research to include in the prompt
            research_summary = f"Research on {self.topic} with {len(all_files)} sources."
            
            # Collect some sample questions explored
            questions = []
            for result in self.research_log:
                if "question" in result:
                    questions.append(result["question"])
            
            research_summary += f"\n\nQuestions explored: {', '.join(questions[:5])}"
            
            # Create a simpler plan prompt that explicitly asks for valid JSON
            plan_prompt = """
            You are an expert report planner. Based on our research on {topic}, create a detailed
            outline for an academic-quality report.
            
            Research summary:
            {research_summary}
            
            Include:
            1. Executive summary
            2. Introduction with clear problem statement
            3. 5-8 main sections with subsections
            4. Methodology section describing research approach
            5. Results and analysis
            6. Discussion of findings
            7. Conclusions and recommendations
            8. References section
            
            Your response MUST be valid JSON that strictly follows this structure:
            {{
                "title": "Report title",
                "sections": [
                    {{
                        "section_title": "Section title",
                        "subsections": ["Subsection 1", "Subsection 2"],
                        "key_points": ["Key point to address 1", "Key point to address 2"]
                    }}
                ]
            }}
            
            Make sure your JSON is properly formatted with all quotes, brackets, and braces.
            DO NOT include anything else outside the JSON structure.
            """
            
            try:
                # First clear memory
                gc.collect()
                
                # Create a default fallback plan in case all else fails
                fallback_plan = {
                    "title": f"Ethical Considerations in {self.topic}",
                    "sections": [
                        {
                            "section_title": "Executive Summary",
                            "subsections": [],
                            "key_points": ["Summary of key findings"]
                        },
                        {
                            "section_title": "Introduction",
                            "subsections": ["Background", "Problem Statement"],
                            "key_points": ["Introduce the topic", "Explain the importance"]
                        },
                        {
                            "section_title": "Methodology",
                            "subsections": ["Research Approach"],
                            "key_points": ["How the research was conducted"]
                        },
                        {
                            "section_title": "Results and Analysis",
                            "subsections": ["Key Findings"],
                            "key_points": ["Main research findings"]
                        },
                        {
                            "section_title": "Discussion",
                            "subsections": ["Implications"],
                            "key_points": ["Interpretation of findings"]
                        },
                        {
                            "section_title": "Conclusions",
                            "subsections": ["Recommendations"],
                            "key_points": ["Main conclusions", "Next steps"]
                        },
                        {
                            "section_title": "References",
                            "subsections": [],
                            "key_points": ["Sources cited in the report"]
                        }
                    ]
                }
                
                # Try to get a response from the LLM
                response = self.llm.invoke(plan_prompt.format(
                    topic=self.topic, 
                    research_summary=research_summary
                ))
                
                print("Received plan from LLM, parsing JSON...")
                
                # Various attempts to extract valid JSON
                try:
                    # First attempt - try to parse directly
                    plan = json.loads(response)
                    print("Successfully parsed JSON response")
                except json.JSONDecodeError as e1:
                    print(f"Direct JSON parsing failed: {str(e1)}")
                    
                    # Second attempt - try to extract JSON from text
                    import re
                    json_pattern = r'(\{[\s\S]*\})'
                    matches = re.findall(json_pattern, response)
                    
                    if matches:
                        try:
                            plan = json.loads(matches[0])
                            print("Successfully extracted and parsed JSON")
                        except json.JSONDecodeError as e2:
                            print(f"Extracted JSON parsing failed: {str(e2)}")
                            plan = fallback_plan
                    else:
                        print("No JSON pattern found in response")
                        plan = fallback_plan
                
                # Validate the plan's structure
                if "title" not in plan or "sections" not in plan or not isinstance(plan["sections"], list):
                    print("Plan structure is invalid, using fallback plan")
                    plan = fallback_plan
                
                # Save the plan
                with open(f"{self.research_dir}/report_plan.json", "w") as f:
                    json.dump(plan, f, indent=2)
                
                # Also save the raw response for debugging
                with open(f"{self.research_dir}/report_plan_raw.txt", "w") as f:
                    f.write(str(response))
                
                return plan
                
            except Exception as e:
                print(f"Error in plan_report: {str(e)}")
                print("Using fallback report plan")
                
                # Save the fallback plan
                with open(f"{self.research_dir}/report_plan_fallback.json", "w") as f:
                    json.dump(fallback_plan, f, indent=2)
                
                return fallback_plan
        else:
            # With embeddings, use RAG as before
            # Query the vector DB for insights to help planning
            retriever = self.db.as_retriever(search_kwargs={"k": 10})  # Reduced from 15 for CPU efficiency
            
            report_plan_prompt = ChatPromptTemplate.from_template("""
            You are an expert report planner. Based on our comprehensive research on {topic}, and the information provided below,
            create a detailed outline for an academic-quality report.
            
            Research context:
            {context}
            
            Include:
            1. Executive summary
            2. Introduction with clear problem statement
            3. 5-8 main sections with subsections
            4. Methodology section describing research approach
            5. Results and analysis
            6. Discussion of findings
            7. Conclusions and recommendations
            8. References section (using academic citation format)
            
            Format your response as JSON with the following structure:
            {{
                "title": "Report title",
                "sections": [
                    {{
                        "section_title": "Section title",
                        "subsections": ["Subsection 1", "Subsection 2"],
                        "key_points": ["Key point to address 1", "Key point to address 2"]
                    }}
                ]
            }}
            """)
            
            # Create RAG chain with memory optimization
            try:
                # First clear memory
                gc.collect()
                
                # Monitor memory usage
                mem = psutil.virtual_memory()
                print(f"Memory before RAG: {mem.percent}% used")
                
                # Create and run RAG chain with CPU optimization
                rag_chain = (
                    {"context": retriever, "topic": RunnablePassthrough()}
                    | report_plan_prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                result = rag_chain.invoke(self.topic)
                
                # Force garbage collection after RAG operation
                gc.collect()
                mem = psutil.virtual_memory()
                print(f"Memory after RAG: {mem.percent}% used")
                
                # Parse the JSON result
                try:
                    plan = json.loads(result)
                    # Save the plan
                    with open(f"{self.research_dir}/report_plan.json", "w") as f:
                        json.dump(plan, f, indent=2)
                    return plan
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    with open(f"{self.research_dir}/report_plan_raw.txt", "w") as f:
                        f.write(result)
                    return {"error": "Failed to parse report plan", "raw_plan": result}
            except Exception as e:
                print(f"Error in plan_report: {str(e)}")
                return {"error": "Error generating report plan", "details": str(e)}
    
    def generate_section(self, section_info):
        """Generate a section using either embeddings RAG or file-based approach"""
        # Clear memory before starting
        gc.collect()
        mem_start = psutil.virtual_memory()
        print(f"Memory before section generation: {mem_start.percent}% used")
        
        if not self.use_embeddings:
            # Without embeddings, use a simpler approach by loading relevant files
            section_title = section_info.get("section_title", "")
            key_points = section_info.get("key_points", [])
            
            # Collect all research files
            all_files = []
            results_dir = f"{self.research_dir}/results"
            if os.path.exists(results_dir):
                all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.txt')]
            
            # Read files to find content relevant to this section
            section_content = []
            keywords = [
                section_title.lower(), 
                *[kp.lower() for kp in key_points]
            ]
            
            relevant_content = []
            for file_path in all_files[:50]:  # Limit to 50 files max
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if content is relevant to this section
                    is_relevant = False
                    for keyword in keywords:
                        if keyword and len(keyword) > 3 and keyword in content.lower():
                            is_relevant = True
                            break
                    
                    if is_relevant:
                        # Extract metadata from first line if present
                        metadata = {}
                        if content.startswith("METADATA:"):
                            meta_end = content.find("\n\n")
                            if meta_end > 0:
                                meta_str = content[9:meta_end]
                                try:
                                    metadata = json.loads(meta_str)
                                    content = content[meta_end+2:]
                                except:
                                    pass
                        
                        # Add snippet with metadata
                        relevant_content.append({
                            "content": content[:1000],  # First 1000 chars
                            "source": file_path,
                            "metadata": metadata
                        })
                        
                        # Limit total content for memory reasons
                        if len(relevant_content) >= 10:
                            break
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
            
            # Prepare context for the prompt
            context_text = "\n\n---\n\n".join([
                f"Source: {item.get('metadata', {}).get('tool', 'Unknown')}\n"
                f"Query: {item.get('metadata', {}).get('query', 'Unknown')}\n\n"
                f"{item['content']}" 
                for item in relevant_content
            ])
            
            if not context_text:
                context_text = "No specific research data available for this section."
            
            section_prompt = ChatPromptTemplate.from_template("""
            You are writing a section of an academic research report on {topic}.
            
            Section to write: {section_title}
            Key points to address: {key_points}
            
            Research information:
            {context}
            
            Write a comprehensive, well-structured section that:
            1. Uses proper academic language and tone
            2. Includes specific facts, figures, and examples from the research
            3. Properly cites sources
            4. Provides critical analysis, not just summary
            5. Makes logical connections between ideas
            6. Is approximately 800-1200 words in length
            
            The section should stand on its own as a quality piece of academic writing.
            """)
            
            try:
                result = section_prompt.format(
                    topic=self.topic,
                    section_title=section_title,
                    key_points=", ".join(key_points),
                    context=context_text
                )
                
                # Generate with the LLM with retry logic for rate limits
                max_retries = 3
                retry_count = 0
                retry_delay = 10  # seconds
                
                while retry_count < max_retries:
                    try:
                        content = self.llm.invoke(result)
                        break  # Success, exit the retry loop
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                            retry_count += 1
                            if retry_count < max_retries:
                                retry_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                                print(f"Rate limit hit. Retrying in {retry_time} seconds... (Attempt {retry_count}/{max_retries})")
                                time.sleep(retry_time)
                            else:
                                print(f"Max retries reached. Using placeholder content.")
                                content = f"[Rate limit reached. This section ({section_title}) needs to be regenerated.]"
                        else:
                            # Other error, not rate limit related
                            print(f"Error generating content: {str(e)}")
                            content = f"[Error generating content: {str(e)}. This section ({section_title}) needs to be regenerated.]"
                            break
                
                # Force memory cleanup after generation
                gc.collect()
                mem_end = psutil.virtual_memory()
                print(f"Memory after section generation: {mem_end.percent}% used")
                print(f"Memory change: {mem_end.percent - mem_start.percent}%")
                
                return {
                    "section_title": section_title,
                    "content": content
                }
            except Exception as e:
                print(f"Error generating section {section_title}: {str(e)}")
                return {
                    "section_title": section_title,
                    "content": f"Error generating content: {str(e)}\n\nPlease regenerate this section.",
                    "error": str(e)
                }
        else:
            # With embeddings, use RAG as before
            # Use fewer documents for retrieval on CPU
            retriever = self.db.as_retriever(search_kwargs={"k": 8})  # Reduced from 10
            
            section_prompt = ChatPromptTemplate.from_template("""
            You are writing a section of an academic research report on {topic}.
            
            Section to write: {section_title}
            Key points to address: {key_points}
            
            Research information:
            {context}
            
            Write a comprehensive, well-structured section that:
            1. Uses proper academic language and tone
            2. Includes specific facts, figures, and examples from the research
            3. Properly cites sources using academic citation format
            4. Provides critical analysis, not just summary
            5. Makes logical connections between ideas
            6. Is approximately 800-1200 words in length
            
            The section should stand on its own as a quality piece of academic writing.
            """)
            
            # Create RAG chain for section generation with CPU optimization
            try:
                rag_chain = (
                    {
                        "context": retriever, 
                        "topic": lambda _: self.topic,
                        "section_title": lambda _: section_info.get("section_title", ""),
                        "key_points": lambda _: ", ".join(section_info.get("key_points", []))
                    }
                    | section_prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                # Generate the section
                result = rag_chain.invoke({})
                
                # Force memory cleanup after generation
                gc.collect()
                mem_end = psutil.virtual_memory()
                print(f"Memory after section generation: {mem_end.percent}% used")
                print(f"Memory change: {mem_end.percent - mem_start.percent}%")
                
                return {
                    "section_title": section_info.get("section_title", ""),
                    "content": result
                }
            except Exception as e:
                print(f"Error generating section {section_info.get('section_title', '')}: {str(e)}")
                return {
                    "section_title": section_info.get("section_title", ""),
                    "content": f"Error generating content: {str(e)}\n\nPlease regenerate this section.",
                    "error": str(e)
                }
    
    def print_memory_usage(self):
        """Print current memory usage"""
        mem = psutil.virtual_memory()
        print(f"Memory usage: {mem.percent}% of total ({mem.used / 1024 / 1024:.1f}MB used, {mem.available / 1024 / 1024:.1f}MB available)")
    
    def generate_report(self, report_plan):
        """Generate the full report section by section with CPU optimization"""
        sections = report_plan.get("sections", [])
        title = report_plan.get("title", f"Research Report: {self.topic}")
        
        # Check if a partial report already exists
        partial_report_path = f"{self.research_dir}/final_report_in_progress.md"
        start_section = 0
        full_report = [f"# {title}\n\n"]
        section_results = []
        
        if os.path.exists(partial_report_path):
            print(f"Found partial report. Checking which sections were completed...")
            
            # Check which section files exist
            for i in range(len(sections)):
                section_file = f"{self.research_dir}/report_section_{i+1}.md"
                if os.path.exists(section_file):
                    print(f"Section {i+1} already generated")
                    # Load the completed section
                    with open(section_file, 'r', encoding='utf-8') as f:
                        section_content = f.read()
                    
                    full_report.append(section_content)
                    section_results.append({
                        "section_title": sections[i].get("section_title", ""),
                        "content": "Content loaded from file"  # We don't need the actual content in memory
                    })
                    start_section = i + 1
                else:
                    break
            
            print(f"Resuming from section {start_section+1}/{len(sections)}")
        
        # Generate each section one by one, starting from where we left off
        for i in range(start_section, len(sections)):
            section = sections[i]
            try:
                # Monitor memory before section generation
                self.print_memory_usage()
                print(f"\n\nGenerating section {i+1}/{len(sections)}: {section.get('section_title', 'Untitled Section')}")
                
                section_result = self.generate_section(section)
                section_content = f"## {section_result['section_title']}\n\n{section_result['content']}\n\n"
                
                # Add subsections if they exist
                if "subsections" in section and section["subsections"]:
                    for j, subsection in enumerate(section["subsections"]):
                        # Force garbage collection between subsections
                        gc.collect()
                        
                        print(f"  Generating subsection {j+1}/{len(section['subsections'])}: {subsection}")
                        subsection_info = {
                            "section_title": subsection,
                            "key_points": section.get("key_points", [])
                        }
                        subsection_result = self.generate_section(subsection_info)
                        section_content += f"### {subsection_result['section_title']}\n\n{subsection_result['content']}\n\n"
                
                full_report.append(section_content)
                section_results.append(section_result)
                
                # Save each section individually
                with open(f"{self.research_dir}/report_section_{i+1}.md", "w") as f:
                    f.write(section_content)
                
                # Generate intermediate complete report
                intermediate_report = "\n".join(full_report)
                with open(f"{self.research_dir}/final_report_in_progress.md", "w") as f:
                    f.write(intermediate_report)
                
                print(f"Completed section {i+1}/{len(sections)}")
                
                # Force garbage collection after each section
                gc.collect()
                self.print_memory_usage()
                
                # Add delay between sections to avoid rate limits
                if i < len(sections) - 1:
                    print("Waiting 5 seconds before next section to avoid rate limits...")
                    time.sleep(5)
                
            except Exception as e:
                error_section = f"## Error generating section: {section.get('section_title', '')}\n\n{str(e)}\n\n"
                full_report.append(error_section)
                section_results.append({"error": str(e), "section": section})
                print(f"Error generating section {i+1}: {str(e)}")
                
                # Save what we have so far
                intermediate_report = "\n".join(full_report)
                with open(f"{self.research_dir}/final_report_in_progress.md", "w") as f:
                    f.write(intermediate_report)
                
                # Wait before trying the next section (if rate limit hit)
                time.sleep(5)
        
        # Combine all sections
        complete_report = "\n".join(full_report)
        
        # Save the complete report
        with open(f"{self.research_dir}/final_report.md", "w") as f:
            f.write(complete_report)
        
        return {
            "report_title": title,
            "report_path": f"{self.research_dir}/final_report.md",
            "section_count": len(section_results)
        }
        
    def continue_report_generation(self):
        """Continue report generation from where it left off"""
        print(f"Continuing report generation for topic: {self.topic}")
        self.print_memory_usage()
        
        # First load the report plan
        report_plan_path = f"{self.research_dir}/report_plan.json"
        if os.path.exists(report_plan_path):
            try:
                with open(report_plan_path, 'r') as f:
                    report_plan = json.load(f)
                print(f"Loaded report plan with {len(report_plan.get('sections', []))} sections")
            except Exception as e:
                print(f"Error loading report plan: {str(e)}")
                # Create a default report plan
                report_plan = {
                    "title": f"Research Report: {self.topic}",
                    "sections": [
                        {"section_title": "Executive Summary", "subsections": [], "key_points": ["Overview of findings"]},
                        {"section_title": "Introduction", "subsections": ["Background", "Objectives"], "key_points": ["Introduce the topic", "Research questions"]},
                        {"section_title": "Methodology", "subsections": ["Research Approach", "Data Sources"], "key_points": ["How the research was conducted"]},
                        {"section_title": "Results and Analysis", "subsections": ["Key Findings", "Data Analysis"], "key_points": ["Main research findings"]},
                        {"section_title": "Discussion", "subsections": ["Implications", "Limitations"], "key_points": ["Interpretation of findings"]},
                        {"section_title": "Conclusions and Recommendations", "subsections": ["Summary", "Future Directions"], "key_points": ["Main conclusions", "Recommendations for action"]},
                        {"section_title": "References", "subsections": [], "key_points": ["Sources cited in the report"]}
                    ]
                }
        else:
            print("No report plan found. Creating default plan.")
            # Create a default report plan
            report_plan = {
                "title": f"Research Report: {self.topic}",
                "sections": [
                    {"section_title": "Executive Summary", "subsections": [], "key_points": ["Overview of findings"]},
                    {"section_title": "Introduction", "subsections": ["Background", "Objectives"], "key_points": ["Introduce the topic", "Research questions"]},
                    {"section_title": "Methodology", "subsections": ["Research Approach", "Data Sources"], "key_points": ["How the research was conducted"]},
                    {"section_title": "Results and Analysis", "subsections": ["Key Findings", "Data Analysis"], "key_points": ["Main research findings"]},
                    {"section_title": "Discussion", "subsections": ["Implications", "Limitations"], "key_points": ["Interpretation of findings"]},
                    {"section_title": "Conclusions and Recommendations", "subsections": ["Summary", "Future Directions"], "key_points": ["Main conclusions", "Recommendations for action"]},
                    {"section_title": "References", "subsections": [], "key_points": ["Sources cited in the report"]}
                ]
            }
        
        # Generate the report (this will automatically continue from where it left off)
        print("Continuing report generation from where it left off...")
        report_result = self.generate_report(report_plan)
        print(f"Report generation complete. Report saved to: {report_result['report_path']}")
        
        return report_result
    
    def run_full_workflow(self):
        """Run the complete research and report generation workflow with CPU optimization"""
        try:
            print(f"Starting research workflow on topic: {self.topic}")
            self.print_memory_usage()
            
            # Step 1: Plan the research
            print("Planning research...")
            research_plan = self.plan_research()
            
            # Always continue with research even if the plan has errors
            if "error" in research_plan:
                print(f"Warning: Issue in research planning: {research_plan.get('error', 'Unknown error')}")
                print("Continuing with available research plan...")
            
            # Force garbage collection between steps
            gc.collect()
            self.print_memory_usage()
            
            # Step 2: Execute the research plan
            print("Executing research plan...")
            try:
                research_results = self.execute_research(research_plan)
                print(f"Research complete. Added {research_results['total_results']} results to vector database.")
            except Exception as e:
                print(f"Error during research execution: {str(e)}")
                research_results = {"total_results": 0, "research_log": []}
                print("Continuing with report generation despite research errors...")
            
            # Force garbage collection between steps
            gc.collect()
            self.print_memory_usage()
            
            # Step 3: Plan the report structure
            print("Planning report structure...")
            report_plan = self.plan_report()
            
            # Always continue with report generation even if the plan has errors
            if "error" in report_plan:
                print(f"Warning: Issue in report planning: {report_plan.get('error', 'Unknown error')}")
                print("Using default report structure...")
                
                # Create a default report plan
                report_plan = {
                    "title": f"Research Report: {self.topic}",
                    "sections": [
                        {
                            "section_title": "Executive Summary",
                            "subsections": [],
                            "key_points": ["Overview of findings"]
                        },
                        {
                            "section_title": "Introduction",
                            "subsections": ["Background", "Objectives"],
                            "key_points": ["Introduce the topic", "Research questions"]
                        },
                        {
                            "section_title": "Methodology",
                            "subsections": ["Research Approach", "Data Sources"],
                            "key_points": ["How the research was conducted"]
                        },
                        {
                            "section_title": "Results and Analysis",
                            "subsections": ["Key Findings", "Data Analysis"],
                            "key_points": ["Main research findings"]
                        },
                        {
                            "section_title": "Discussion",
                            "subsections": ["Implications", "Limitations"],
                            "key_points": ["Interpretation of findings"]
                        },
                        {
                            "section_title": "Conclusions and Recommendations",
                            "subsections": ["Summary", "Future Directions"],
                            "key_points": ["Main conclusions", "Recommendations for action"]
                        },
                        {
                            "section_title": "References",
                            "subsections": [],
                            "key_points": ["Sources cited in the report"]
                        }
                    ]
                }
            
            # Force garbage collection between steps
            gc.collect()
            self.print_memory_usage()
            
            # Step 4: Generate the report
            print("Generating final report...")
            report_result = self.generate_report(report_plan)
            print(f"Report generation complete. Report saved to: {report_result['report_path']}")
            
            # Final cleanup
            gc.collect()
            self.print_memory_usage()
            
            return {
                "topic": self.topic,
                "session_id": self.session_id,
                "research_dir": self.research_dir,
                "research_results": research_results,
                "report_info": report_result
            }
        except Exception as e:
            print(f"Critical error in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": "Workflow execution failed", "details": str(e)}

# Example usage - uncomment these lines to run the workflow
workflow = ResearchWorkflow(topic="Artificial Intelligence Ethics in Healthcare")
try:
    # Set parameters for this run
    workflow.use_embeddings = False  # Set to False for much faster processing
    workflow.search_limit = 10      # Limit to 10 searches for quick results
    
    # Check for the most recent research directory to continue from
    research_dirs = []
    if os.path.exists("research_sessions"):
        research_dirs = [os.path.join("research_sessions", d) for d in os.listdir("research_sessions") if os.path.isdir(os.path.join("research_sessions", d))]
    
    if research_dirs:
        # Find the most recent directory by creation time
        most_recent = max(research_dirs, key=os.path.getctime)
        
        if os.path.exists(f"{most_recent}/final_report_in_progress.md"):
            print(f"Found partial research in {most_recent}")
            print("Continue from previous session? (y/n)")
            # Check if this script is being executed directly (not imported)
            if __name__ == "__main__":
                continue_prev = input().strip().lower()
                if continue_prev == 'y':
                    # Set the research directory to the most recent one
                    workflow.research_dir = most_recent
                    # Get session ID from the directory name
                    workflow.session_id = os.path.basename(most_recent)
                    # Continue from where we left off
                    result = workflow.continue_report_generation()
                    print(f"Report continuation completed. Report saved at: {result['report_path']}")
                    # Exit early
                    import sys
                    sys.exit()
    
    print(f"Starting workflow with embeddings={'Enabled' if workflow.use_embeddings else 'Disabled'}, "
          f"search_limit={workflow.search_limit}...")
    
    result = workflow.run_full_workflow()
    print(f"Workflow completed with result: {result.get('research_dir', 'No research directory')}")
    print(f"To increase searches, change SEARCH_QUERIES_LIMIT or workflow.search_limit to desired value")
    print(f"To enable/disable embeddings, change USE_EMBEDDINGS or workflow.use_embeddings")
except Exception as e:
    print(f"Error running workflow: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Remind the user they can continue from where they left off
    print("\nYou can continue report generation from where it left off by running:")
    print("python -c \"from tools import workflow; workflow.research_dir = 'LATEST_DIR'; workflow.continue_report_generation()\"")
    print("Replace LATEST_DIR with the path displayed above")
