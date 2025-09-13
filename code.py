import os
import pandas as pd
import numpy as np
import json
import faiss
import openai
import streamlit as st
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalRAGChatbotSimple:

    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key   
        self.faiss_index = None
        self.knowledge_base: List[Dict] = []
        self.embeddings: np.ndarray | None = None

    def load_dataset(self, file_path: str) -> List[Dict]:
        """Load medical FAQ dataset from CSV or JSON file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                if 'question' in df.columns and 'answer' in df.columns:
                    data = df[['question', 'answer']].to_dict('records')
                else:
                    cols = df.columns.tolist()
                    data = []
                    for _, row in df.iterrows():
                        data.append({
                            'question': row[cols[0]],
                            'answer': row[cols[1]]
                        })
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")

            logger.info(f"Loaded {len(data)} FAQ entries from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def preprocess_dataset(self, data: List[Dict]) -> List[str]:
        """Preprocess dataset into chunks of question + answer."""
        chunks: List[str] = []
        self.knowledge_base = []

        for item in data:
            question = item.get('question', '') or ''
            answer = item.get('answer', '') or ''
            chunk = f"Question: {question}\nAnswer: {answer}"
            chunks.append(chunk)
            self.knowledge_base.append({
                'question': question,
                'answer': answer,
                'chunk': chunk
            })

        logger.info(f"Preprocessed {len(chunks)} chunks")
        return chunks

    def create_embeddings_openai(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI's embedding model."""
        logger.info("Creating embeddings using OpenAI embeddings API...")
        embeddings: List[List[float]] = []
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                logger.info(f"Processed {len(embeddings)}/{len(chunks)} embeddings")
            except Exception as e:
                logger.error(f"Error creating embeddings for batch starting at {i}: {e}")
                return self.create_simple_embeddings(chunks)

        embeddings_array = np.array(embeddings, dtype=np.float32)  
        self.embeddings = embeddings_array
        logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

    def create_simple_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Fallback: simple TF-IDF embeddings if OpenAI fails (returns float32)."""
        logger.info("Creating simple TF-IDF embeddings as fallback...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        embeddings = vectorizer.fit_transform(chunks).toarray()
        embeddings_array = np.array(embeddings, dtype=np.float32)  
        self.embeddings = embeddings_array
        logger.info(f"Created fallback embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

    def build_vector_index(self, embeddings: np.ndarray):
        """Build FAISS vector index."""
        if embeddings is None or len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be a 2D numpy array")

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")

    def setup_knowledge_base(self, file_path: str):
        """Load, preprocess, embed and index the dataset."""
        data = self.load_dataset(file_path)
        chunks = self.preprocess_dataset(data)
        embeddings = self.create_embeddings_openai(chunks)
        self.build_vector_index(embeddings)
        logger.info("Knowledge base setup complete!")

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant context for a query."""
        if self.faiss_index is None:
            raise ValueError("Knowledge base not initialized. Call setup_knowledge_base first.")

        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = np.array([response['data'][0]['embedding']], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Embedding creation for query failed: {e}. Using simple similarity fallback.")
            return self.simple_similarity_search(query, top_k)

        faiss.normalize_L2(query_embedding)
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        relevant_context: List[Dict] = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.knowledge_base) and idx != -1:
                context = self.knowledge_base[idx].copy()
                context['similarity_score'] = float(score)
                context['rank'] = i + 1
                relevant_context.append(context)

        return relevant_context

    def simple_similarity_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Fallback similarity search using Jaccard keyword matching."""
        query_words = set(query.lower().split())
        scores = []
        for kb_item in self.knowledge_base:
            text = (kb_item.get('question', '') + " " + kb_item.get('answer', '')).lower()
            text_words = set(text.split())
            intersection = len(query_words.intersection(text_words))
            union = len(query_words.union(text_words))
            score = intersection / union if union > 0 else 0.0
            scores.append(score)

        if not scores:
            return []

        top_indices = np.argsort(scores)[-top_k:][::-1]
        relevant_context = []
        for i, idx in enumerate(top_indices):
            context = self.knowledge_base[int(idx)].copy()
            context['similarity_score'] = float(scores[int(idx)])
            context['rank'] = i + 1
            relevant_context.append(context)
        return relevant_context

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using OpenAI LLM with retrieved context."""
        context_text = ""
        for i, ctx in enumerate(context, 1):
            context_text += f"\nContext {i}:\n{ctx['chunk']}\n"

        system_prompt = """You are a helpful medical FAQ assistant. Use the provided context to answer user questions about medical topics. 
- Provide accurate, clear, and natural answers based on the context
- If the context doesn't contain relevant information, say so politely
- Always recommend consulting healthcare professionals for serious concerns
- Keep responses concise but informative
- Do not provide diagnosis or replace professional medical advice"""

        user_prompt = f"""Context from medical FAQ database:
{context_text}

User Question: {query}

Please provide a helpful answer based on the context above."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            # robust access
            answer = response['choices'][0]['message']['content']
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def chat(self, query: str, top_k: int = 3) -> Dict:
        """Main chat function combining retrieval + generation."""
        try:
            context = self.retrieve_relevant_context(query, top_k)
            response = self.generate_response(query, context)
            return {'query': query, 'response': response, 'context': context, 'success': True}
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {'query': query, 'response': f"Error: {str(e)}", 'context': [], 'success': False}


def main():
    st.set_page_config(page_title="Medical FAQ Chatbot", layout="wide")
    st.title("Medical FAQ Chatbot")
    st.markdown("Ask questions about medical topics and get answers based on our FAQ database.")

    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        uploaded_file = st.file_uploader("Upload Medical FAQ Dataset", type=['csv', 'json'])
        if st.button("Initialize Chatbot"):
            if not openai_api_key:
                st.error("Please provide OpenAI API key")
            elif not uploaded_file:
                st.error("Please upload a dataset file")
            else:
                with st.spinner("Initializing chatbot..."):
                    try:
                        file_path = f"temp_{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        chatbot = MedicalRAGChatbotSimple(openai_api_key)
                        chatbot.setup_knowledge_base(file_path)
                        st.session_state['chatbot'] = chatbot
                        st.session_state['initialized'] = True
                        os.remove(file_path)
                        st.success("Chatbot initialized successfully!")
                    except Exception as e:
                        st.error(f"Error initializing chatbot: {e}")

    if 'initialized' in st.session_state and st.session_state['initialized']:
        chatbot: MedicalRAGChatbotSimple = st.session_state['chatbot']
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_query = st.text_input("Ask a medical question:",
                                   placeholder="e.g., What are the early symptoms of diabetes?",
                                   key="user_input")
        if st.button("Send") or (user_query and st.session_state.get('last_query') != user_query):
            if user_query.strip():
                st.session_state['last_query'] = user_query
                with st.spinner("Searching knowledge base..."):
                    result = chatbot.chat(user_query)
                    st.session_state['chat_history'].append({
                        'query': user_query,
                        'response': result['response'],
                        'context': result['context']
                    })
        if st.session_state['chat_history']:
            st.markdown("---")
            st.header("Chat History")
            for i, chat in enumerate(reversed(st.session_state['chat_history'])):
                with st.expander(f"Q: {chat['query'][:50]}...", expanded=(i == 0)):
                    st.markdown(f"**Question:** {chat['query']}")
                    st.markdown(f"**Answer:** {chat['response']}")
                    if chat['context']:
                        st.markdown("**Relevant Context:**")
                        for j, ctx in enumerate(chat['context']):
                            st.markdown(f"*Context {j+1} (Score: {ctx['similarity_score']:.3f}):*")
                            st.markdown(f"- Q: {ctx['question']}")
                            st.markdown(f"- A: {ctx['answer']}")
    else:
        st.info("Please configure your API key and upload a dataset in the sidebar to get started.")

if __name__ == "__main__":
    main()
