# Medical FAQ RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer medical questions using a knowledge base of medical FAQs. The system retrieves relevant context from a medical FAQ database and generates natural, accurate responses using OpenAI's language models.

## Features

- **RAG Pipeline**: Implements complete Retrieval-Augmented Generation workflow
- **Vector Search**: Uses FAISS for efficient similarity search with sentence embeddings
- **OpenAI Integration**: Leverages GPT-3.5-turbo for natural response generation
- **Dual Interface**: Both Streamlit web app
- **Flexible Input**: Supports CSV and JSON dataset formats
- **Context Awareness**: Shows relevant context and similarity scores
- **Medical Focus**: Specifically designed for medical FAQ responses

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

### Installation

**Prepare your dataset:**
   - Use the provided `faqs.csv` for testing
   - Or download a medical FAQ dataset from Kaggle
   - Ensure your CSV has 'question' and 'answer' columns

### Usage

#### Streamlit Web App 

1. **Launch the Streamlit app:**
   ```bash
   streamlit run medical_rag_chatbot.py
   ```

2. **Configure the chatbot:**
   - Enter your OpenAI API key in the sidebar
   - Upload your medical FAQ dataset (CSV or JSON)
   - Click "Initialize Chatbot"

3. **Start chatting:**
   - Ask medical questions in the text input
   - View responses with relevant context
   - Check chat history in expandable sections

## Dataset Format

### CSV Format
```csv
question,answer
"What are the symptoms of diabetes?","Symptoms include increased thirst, frequent urination..."
"Can children take aspirin?","Children should generally avoid aspirin due to..."
```

### JSON Format
```json
[
  {
    "question": "What are the symptoms of diabetes?",
    "answer": "Symptoms include increased thirst, frequent urination..."
  }
]
```

## 🔧 Technical Architecture

### RAG Pipeline Components

1. **Data Preprocessing**
   - Loads medical FAQ dataset from CSV/JSON
   - Combines questions and answers into searchable chunks
   - Handles text cleaning and formatting

2. **Embedding Generation**
   - Uses SentenceTransformers (`all-MiniLM-L6-v2`) for semantic embeddings
   - Creates vector representations of all FAQ entries
   - Optimized for medical domain similarity

3. **Vector Database**
   - FAISS (Facebook AI Similarity Search) for efficient similarity search
   - Cosine similarity for finding relevant context
   - Fast retrieval even with large datasets

4. **Response Generation**
   - OpenAI GPT-3.5-turbo for natural language generation
   - Context-aware prompting with retrieved FAQ entries
   - Medical-specific system prompts for accurate responses

## Example Queries

Try these sample questions with the chatbot:

- "What are the early symptoms of diabetes?"
- "Can children take paracetamol?"
- "What foods are good for heart health?"

## Design Choices

### Why These Technologies?

1. **FAISS**: Chosen for its speed and efficiency in similarity search, crucial for real-time chatbot responses
2. **SentenceTransformers**: Provides high-quality semantic embeddings optimized for sentence similarity
3. **OpenAI GPT-3.5**: Balances response quality with cost-effectiveness
4. **Streamlit**: Rapid prototyping for user-friendly interface
5. **Modular Design**: Separate concerns for easy maintenance and testing

### RAG vs Fine-tuning

- **RAG Benefits**: No retraining needed, easy to update knowledge base, transparent reasoning
- **Use Case Fit**: Perfect for FAQ systems where source attribution is important
