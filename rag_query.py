#!/usr/bin/env python3
"""
RAG Query Script
Uses the prepared vector database to answer questions from Confluence content
"""

import os
import yaml
import requests
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGQuery:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG query system"""
        self.config = self._load_config(config_path)
        self.vector_db_path = self.config['vector_db']['path']
        self.embedding_model = SentenceTransformer(self.config['vector_db']['embedding_model'])
        self.top_k = self.config['rag']['top_k']
        self.similarity_threshold = self.config['rag']['similarity_threshold']
        
        # Load vector database
        self.index, self.metadata_df, self.chunks_df = self._load_vector_database()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _load_vector_database(self) -> Tuple[faiss.Index, pd.DataFrame, pd.DataFrame]:
        """Load the FAISS index and metadata"""
        index_path = os.path.join(self.vector_db_path, 'faiss_index.bin')
        metadata_path = os.path.join(self.vector_db_path, 'metadata.csv')
        chunks_path = os.path.join(self.vector_db_path, 'chunks.csv')
        
        if not all(os.path.exists(p) for p in [index_path, metadata_path, chunks_path]):
            raise FileNotFoundError("Vector database files not found. Please run the preparation script first.")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata and chunks
        metadata_df = pd.read_csv(metadata_path)
        chunks_df = pd.read_csv(chunks_path)
        
        logger.info(f"Loaded vector database with {len(chunks_df)} chunks")
        return index, metadata_df, chunks_df
    
    def search_similar_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Search for similar chunks using hybrid approach: semantic + fallback keyword matching"""
        # Create query embedding for semantic search
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        similarities, indices = self.index.search(
            query_embedding.astype('float32'), 
            self.top_k
        )
        
        semantic_results = []
        for i, (similarity, chunk_idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= self.similarity_threshold:
                chunk_data = self.chunks_df.iloc[chunk_idx]
                metadata = self.metadata_df.iloc[chunk_idx]
                
                semantic_results.append({
                    'chunk_id': int(chunk_idx),
                    'text': chunk_data['text'],
                    'similarity': float(similarity),
                    'page_title': metadata['title'],
                    'page_id': metadata['page_id'],
                    'chunk_index': metadata['chunk_index'],
                    'total_chunks': metadata['total_chunks'],
                    'search_type': 'semantic'
                })
        
        # Return semantic search results
        return semantic_results
    
    def query_llm(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Query the local LLM with context"""
        llm_config = self.config['llm']
        
        # Prepare context from chunks
        context_text = "\n\n".join([
            f"From '{chunk['page_title']}':\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context from a Confluence wiki.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so. Be specific and reference the source pages when possible.

Answer:"""
        
        # Prepare request for local LLM (Ollama)
        if llm_config['endpoint'].startswith('http://localhost:11434'):
            # Ollama format
            request_data = {
                "model": llm_config['model'],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": llm_config['temperature'],
                    "num_predict": llm_config['max_tokens']
                }
            }
        else:
            # Generic OpenAI-compatible format
            request_data = {
                "model": llm_config['model'],
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": llm_config['temperature'],
                "max_tokens": llm_config['max_tokens']
            }
        
        try:
            response = requests.post(
                llm_config['endpoint'],
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            
            if llm_config['endpoint'].startswith('http://localhost:11434'):
                # Ollama response format
                result = response.json()
                return result.get('response', 'No response from LLM')
            else:
                # Generic OpenAI-compatible response format
                result = response.json()
                return result.get('choices', [{}])[0].get('message', {}).get('content', 'No response from LLM')
                
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"Error querying LLM: {e}"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main method to answer a question using RAG"""
        logger.info(f"Processing question: {question}")
        
        # Search for relevant chunks
        similar_chunks = self.search_similar_chunks(question)
        
        if not similar_chunks:
            return {
                'question': question,
                'answer': 'No relevant information found in the knowledge base.',
                'sources': [],
                'similarity_scores': []
            }
        
        # Query LLM with context
        answer = self.query_llm(question, similar_chunks)
        
        # Prepare sources information
        sources = []
        similarity_scores = []
        
        for chunk in similar_chunks:
            sources.append({
                'page_title': chunk['page_title'],
                'page_id': chunk['page_id'],
                'chunk_index': chunk['chunk_index'],
                'similarity': chunk['similarity']
            })
            similarity_scores.append(chunk['similarity'])
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'similarity_scores': similarity_scores
        }
    
    def interactive_mode(self):
        """Run in interactive mode for continuous Q&A"""
        print("RAG Query System - Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Get answer
                result = self.answer_question(question)
                
                # Display answer
                print(f"\nAnswer: {result['answer']}")
                
                # Display sources
                if result['sources']:
                    print(f"\nSources:")
                    for i, source in enumerate(result['sources']):
                        print(f"  {i+1}. {source['page_title']} (similarity: {source['similarity']:.3f})")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RAG Query System for Confluence Wiki')
    parser.add_argument('--question', '-q', type=str, help='Single question to answer')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        rag_query = RAGQuery(args.config)
        
        if args.question:
            # Single question mode
            result = rag_query.answer_question(args.question)
            
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources:")
                for i, source in enumerate(result['sources']):
                    print(f"  {i+1}. {source['page_title']} (similarity: {source['similarity']:.3f})")
        
        elif args.interactive:
            # Interactive mode
            rag_query.interactive_mode()
        
        else:
            # Default to interactive mode
            rag_query.interactive_mode()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
