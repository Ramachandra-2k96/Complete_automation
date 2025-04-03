# IonIdea Conversation Evaluator

This project creates an automated conversation system with two AI models: an "actor" model that simulates a human website visitor, and a RAG-based (Retrieval Augmented Generation) model that answers questions about IonIdea software company.

## Features

- **Actor LLM**: Simulates different personas (customer, job seeker, sales rep, curious visitor)
- **RAG LLM**: Answers questions by retrieving information from IonIdea documents
- **Conversation Evaluation**: Actor rates responses as "perfect", "good", "moderate", or "bad"
- **CSV Logging**: All conversations are logged with evaluation ratings
- **Realistic Conversation Flow**: The actor asks follow-up questions based on previous responses

## Requirements

- Python 3.8+
- Ollama (for local LLMs)
- LlamaIndex
- ChromaDB
- NLTK
- Other dependencies listed in the imports

## Setup

1. Make sure you have Ollama installed and running with the required models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. Create a directory called `ionideafiles` and place PDF documents about IonIdea in it.

3. Run the script:
   ```
   python conversation_evaluator.py
   ```

4. Enter the number of question-answer exchanges you want to simulate.

## How It Works

1. **Document Loading**: The system loads PDF documents from the `ionideafiles` directory.
2. **RAG Setup**: Documents are indexed and stored in ChromaDB for retrieval.
3. **Actor Selection**: A random persona is selected (customer, job seeker, etc.).
4. **Conversation Loop**:
   - Actor asks a question
   - RAG system answers with relevant retrieved information
   - Actor evaluates the response quality
   - Data is logged to CSV
   - Actor formulates follow-up question
5. **CSV Output**: Results are saved to `conversation_evaluations.csv` with columns:
   - Timestamp
   - Actor Role
   - Question
   - Response
   - Citations
   - Correctness rating

## Example Output

The terminal output will look like this:

```
[INFO] Starting conversation with Actor as Job Seeker
[INFO] The actor will ask 5 questions
--------------------------------------------------------------------------------
[Job Seeker]: What positions are typically available at IonIdea?
[IonIdea Representative]: IonIdea offers a variety of positions across different technical and non-technical departments. Our technical roles include Software Engineers, Full Stack Developers, Data Scientists, QA Engineers, DevOps Engineers, and Cloud Architects. We also have positions in Project Management, Business Analysis, UI/UX Design, and various leadership roles like Technical Leads and Engineering Managers. Additionally, we have openings in support functions like HR, Finance, and Administration. The specific positions available vary based on our current projects and growth needs.
Citations:
- careers_brochure.pdf, Page: 3
- company_overview.pdf, Page: 12
[Evaluation]: good
--------------------------------------------------------------------------------
```

## Files

- `conversation_evaluator.py`: Main script that runs the simulation
- `conversation_evaluations.csv`: CSV output with all conversation data and evaluations
- `README.md`: This documentation file 