#!/usr/bin/env python3
"""
Confluence RAG Preparation Script
Extracts content from Confluence wiki and creates embeddings for RAG system
"""

import os
import yaml
import requests
import json
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import markdown
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfluenceRAG:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the Confluence RAG system"""
        self.config = self._load_config(config_path)
        self.confluence_session = self._create_confluence_session()
        self.embedding_model = SentenceTransformer(self.config['vector_db']['embedding_model'])
        self.vector_db_path = self.config['vector_db']['path']
        self.chunk_size = self.config['vector_db']['chunk_size']
        self.chunk_overlap = self.config['vector_db']['chunk_overlap']
        
        # Create vector database directory
        os.makedirs(self.vector_db_path, exist_ok=True)
        
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
    
    def _create_confluence_session(self) -> requests.Session:
        """Create authenticated session for Confluence API"""
        session = requests.Session()
        
        # Use Bearer token authentication instead of basic auth
        session.headers.update({
            'Authorization': f'Bearer {self.config["confluence"]["api_token"]}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        return session
    
    def get_space_pages(self) -> List[Dict[str, Any]]:
        """Get all pages from the specified Confluence space"""
        base_url = self.config['confluence']['base_url']
        space_key = self.config['confluence']['space_key']
        
        pages = []
        start = 0
        limit = 50
        
        while True:
            # Use REST API with body content expansion - avoid double /wiki/ path
            url = f"{base_url}/rest/api/content"
            params = {
                'spaceKey': space_key,
                'start': start,
                'limit': limit,
                'type': 'page',
                'status': 'current',
                'expand': 'body.storage,version,space'
            }
            
            try:
                response = self.confluence_session.get(url, params=params)
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                if response.status_code != 200:
                    logger.error(f"Error response: {response.text[:500]}")
                    break
                
                # Log response content for debugging
                logger.info(f"Response content preview: {response.text[:200]}")
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get('results'):
                    break
                    
                pages.extend(data['results'])
                start += limit
                
                if len(data['results']) < limit:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching pages: {e}")
                logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
                break
        
        logger.info(f"Retrieved {len(pages)} pages from Confluence")
        return pages
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """Get detailed content of a specific page"""
        base_url = self.config['confluence']['base_url']
        
        # Use REST API with proper expansion for body content
        page_url = f"{base_url}/wiki/rest/api/content/{page_id}"
        params = {
            'expand': 'body.storage,version,space'
        }
        
        try:
            response = self.confluence_session.get(page_url, params=params)
            response.raise_for_status()
            page_data = response.json()
            
            # Extract body content from the response
            body_content = page_data.get('body', {}).get('storage', {}).get('value', '')
            body_format = 'storage'  # Confluence storage format
            
            return {
                'id': page_id,
                'title': page_data.get('title', ''),
                'space_id': page_data.get('space', {}).get('key', ''),
                'body': body_content,
                'body_format': body_format,
                'created': page_data.get('created', ''),
                'updated': page_data.get('version', {}).get('when', '')
            }
            
        except Exception as e:
            logger.error(f"Error fetching page {page_id}: {e}")
            return None
    
    def clean_content(self, content: str, format_type: str) -> str:
        """Clean and normalize content from Confluence with proper structure preservation"""
        if format_type == 'atlas_doc_format':
            # Convert Atlassian Document Format to structured text
            try:
                doc_data = json.loads(content)
                text_content = self._extract_text_from_atlas_doc(doc_data)
            except:
                text_content = content
        elif format_type == 'storage':
            # Convert Confluence storage format to structured text
            text_content = self._extract_text_from_storage_format(content)
        else:
            text_content = content
        
        return text_content.strip()
    
    def _extract_text_from_storage_format(self, html_content: str) -> str:
        """Extract structured text from Confluence storage format HTML"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text_parts = []
        
        # Process all content elements including tables
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'table']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Add heading with clear separation
                text_parts.append(f"\n\n## {element.get_text().strip()}\n")
            elif element.name == 'p':
                # Add paragraph with proper spacing
                text_parts.append(f"\n{element.get_text().strip()}\n")
            elif element.name == 'table':
                # Handle tables properly - this is crucial for delegation tasks
                table_text = self._extract_table_text(element)
                if table_text:
                    text_parts.append(f"\n\n{table_text}\n\n")
            elif element.name in ['ul', 'ol']:
                # Handle lists properly
                list_text = self._extract_list_text(element)
                text_parts.append(f"\n{list_text}\n")
            elif element.name == 'li':
                # Skip individual list items as they're handled by parent
                continue
            elif element.name == 'blockquote':
                # Handle blockquotes
                text_parts.append(f"\n> {element.get_text().strip()}\n")
            elif element.name in ['pre', 'code']:
                # Handle code blocks
                text_parts.append(f"\n```\n{element.get_text().strip()}\n```\n")
        
        # If no structured elements found, fall back to general text extraction
        if not text_parts:
            text_parts.append(soup.get_text())
        
        return '\n'.join(text_parts)
    
    def _extract_table_text(self, table) -> str:
        """Extract structured text from HTML table with enhanced Confluence support"""
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        table_text = []
        
        for i, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            # Determine if this is a header row
            is_header = i == 0 or any(cell.name == 'th' for cell in cells)
            
            # Extract cell content with better text handling
            cell_texts = []
            for cell in cells:
                # Get text content, handling nested elements
                cell_content = cell.get_text(separator=' ', strip=True)
                if cell_content:
                    # Clean up extra whitespace
                    cell_content = ' '.join(cell_content.split())
                    cell_texts.append(cell_content)
                else:
                    # Even empty cells should be represented
                    cell_texts.append("")
            
            if cell_texts:
                if is_header:
                    table_text.append(f"| {' | '.join(cell_texts)} |")
                    # Add separator line for markdown table
                    table_text.append(f"| {' | '.join(['---'] * len(cell_texts))} |")
                else:
                    table_text.append(f"| {' | '.join(cell_texts)} |")
        
        if table_text:
            return '\n'.join(table_text)
        return ""
    
    def _extract_list_text(self, list_element) -> str:
        """Extract structured text from HTML lists"""
        items = list_element.find_all('li', recursive=False)
        list_text = []
        
        for item in items:
            # Get the main text of this list item
            item_text = item.get_text().strip()
            if item_text:
                # Check if this is an ordered or unordered list
                if list_element.name == 'ol':
                    list_text.append(f"1. {item_text}")
                else:
                    list_text.append(f"• {item_text}")
                
                # Handle nested lists
                nested_lists = item.find_all(['ul', 'ol'], recursive=False)
                for nested_list in nested_lists:
                    nested_text = self._extract_list_text(nested_list)
                    if nested_text:
                        # Indent nested list items
                        nested_lines = nested_text.split('\n')
                        indented_lines = [f"  {line}" for line in nested_lines]
                        list_text.extend(indented_lines)
        
        return '\n'.join(list_text)
    
    def _extract_text_from_atlas_doc(self, doc_data: Dict[str, Any]) -> str:
        """Extract structured text from Atlassian Document Format"""
        text_parts = []
        
        def extract_content(obj, level=0):
            if isinstance(obj, dict):
                obj_type = obj.get('type', '')
                
                if obj_type == 'paragraph':
                    content_text = []
                    for content in obj.get('content', []):
                        if content.get('type') == 'text':
                            content_text.append(content.get('text', ''))
                    if content_text:
                        text_parts.append(f"\n{'  ' * level}{''.join(content_text)}\n")
                
                elif obj_type == 'heading':
                    level_num = obj.get('attrs', {}).get('level', 1)
                    content_text = []
                    for content in obj.get('content', []):
                        if content.get('type') == 'text':
                            content_text.append(content.get('text', ''))
                    if content_text:
                        heading_text = ''.join(content_text)
                        text_parts.append(f"\n{'#' * level_num} {heading_text}\n")
                
                elif obj_type == 'table':
                    # Handle tables in ADF
                    table_text = self._extract_adf_table_text(obj)
                    text_parts.append(f"\n\n{table_text}\n\n")
                
                elif obj_type == 'bulletList':
                    # Handle bullet lists in ADF
                    list_text = self._extract_adf_list_text(obj, level)
                    text_parts.append(f"\n{list_text}\n")
                
                elif obj_type == 'orderedList':
                    # Handle ordered lists in ADF
                    list_text = self._extract_adf_ordered_list_text(obj, level)
                    text_parts.append(f"\n{list_text}\n")
                
                else:
                    # Recursively process other content
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            extract_content(value, level)
            
            elif isinstance(obj, list):
                for item in obj:
                    extract_content(item, level)
        
        extract_content(doc_data)
        return '\n'.join(text_parts)
    
    def _extract_adf_table_text(self, table_obj: Dict[str, Any]) -> str:
        """Extract structured text from ADF table"""
        table_text = []
        
        # Get table content
        content = table_obj.get('content', [])
        if not content:
            return ""
        
        for row in content:
            if row.get('type') == 'tableRow':
                row_content = row.get('content', [])
                cell_texts = []
                
                for cell in row_content:
                    if cell.get('type') == 'tableCell':
                        cell_content = cell.get('content', [])
                        cell_text = self._extract_adf_cell_text(cell_content)
                        cell_texts.append(cell_text)
                
                if cell_texts:
                    table_text.append(f"| {' | '.join(cell_texts)} |")
        
        if table_text:
            # Add separator line after first row (assuming it's a header)
            if len(table_text) > 0:
                separator = f"| {' | '.join(['---'] * len(table_text[0].split('|')[1:-1]))} |"
                table_text.insert(1, separator)
        
        return '\n'.join(table_text)
    
    def _extract_adf_cell_text(self, cell_content: List[Dict[str, Any]]) -> str:
        """Extract text from ADF table cell"""
        text_parts = []
        
        for content in cell_content:
            if content.get('type') == 'text':
                text_parts.append(content.get('text', ''))
            elif content.get('type') == 'paragraph':
                # Handle nested paragraphs in cells
                for para_content in content.get('content', []):
                    if para_content.get('type') == 'text':
                        text_parts.append(para_content.get('text', ''))
        
        return ' '.join(text_parts)
    
    def _extract_adf_list_text(self, list_obj: Dict[str, Any], level: int) -> str:
        """Extract text from ADF bullet list"""
        list_text = []
        
        for item in list_obj.get('content', []):
            if item.get('type') == 'listItem':
                item_content = item.get('content', [])
                item_text = self._extract_adf_list_item_text(item_content)
                if item_text:
                    list_text.append(f"{'  ' * level}• {item_text}")
        
        return '\n'.join(list_text)
    
    def _extract_adf_ordered_list_text(self, list_obj: Dict[str, Any], level: int) -> str:
        """Extract text from ADF ordered list"""
        list_text = []
        
        for i, item in enumerate(list_obj.get('content', []), 1):
            if item.get('type') == 'listItem':
                item_content = item.get('content', [])
                item_text = self._extract_adf_list_item_text(item_content)
                if item_text:
                    list_text.append(f"{'  ' * level}{i}. {item_text}")
        
        return '\n'.join(list_text)
    
    def _extract_adf_list_item_text(self, item_content: List[Dict[str, Any]]) -> str:
        """Extract text from ADF list item"""
        text_parts = []
        
        for content in item_content:
            if content.get('type') == 'paragraph':
                for para_content in content.get('content', []):
                    if para_content.get('type') == 'text':
                        text_parts.append(para_content.get('text', ''))
            elif content.get('type') == 'text':
                text_parts.append(content.get('text', ''))
        
        return ' '.join(text_parts)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into meaningful chunks based on content structure with strict size limits"""
        # Split by major sections first (double newlines)
        sections = text.split('\n\n')
        chunks = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is small enough, keep it as one chunk
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Split large sections by single newlines (paragraphs, list items, etc.)
                lines = section.split('\n')
                current_chunk = []
                current_length = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # If adding this line would exceed chunk size, save current chunk
                    if current_length + len(line) > self.chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                        current_length = len(line)
                    else:
                        current_chunk.append(line)
                        current_length += len(line)
                
                # Add the last chunk
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
        
        # Ensure chunks don't exceed maximum size - this is critical for token efficiency
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # For extremely long chunks, split by sentences with strict size control
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # If adding this sentence would exceed chunk size, save current chunk
                    if current_length + len(sentence) > self.chunk_size and current_chunk:
                        final_chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                
                # Add the last chunk
                if current_chunk:
                    final_chunks.append(' '.join(current_chunk))
        
        # Final validation: ensure no chunk exceeds the maximum size
        validated_chunks = []
        for chunk in final_chunks:
            if len(chunk) <= self.chunk_size:
                validated_chunks.append(chunk)
            else:
                # Emergency split for any remaining oversized chunks
                logger.warning(f"Found oversized chunk ({len(chunk)} chars), splitting further")
                # Split by words to ensure we stay under limit
                words = chunk.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > self.chunk_size and current_chunk:
                        validated_chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1  # +1 for space
                
                if current_chunk:
                    validated_chunks.append(' '.join(current_chunk))
        
        # Log chunk statistics for debugging
        chunk_lengths = [len(chunk) for chunk in validated_chunks]
        if chunk_lengths:
            logger.debug(f"Created {len(validated_chunks)} chunks: min={min(chunk_lengths)}, max={max(chunk_lengths)}, mean={sum(chunk_lengths)/len(chunk_lengths):.1f}")
        
        return validated_chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_vector_database(self, documents: List[Dict[str, Any]]) -> None:
        """Build and save the vector database"""
        all_chunks = []
        all_metadata = []
        
        logger.info("Processing documents and creating chunks...")
        
        for doc in tqdm(documents, desc="Processing documents"):
            # Extract body content from the page data
            body_content = doc.get('body', {}).get('storage', {}).get('value', '')
            if not body_content:
                logger.debug(f"Skipping document {doc.get('id', 'unknown')}: no body content")
                continue
                
            # Clean content
            clean_text = self.clean_content(body_content, 'storage')
            if not clean_text:
                logger.debug(f"Skipping document {doc.get('id', 'unknown')}: no clean text after processing")
                continue
            
            logger.debug(f"Document {doc.get('id', 'unknown')} - Title: {doc.get('title', 'unknown')}")
            logger.debug(f"Body format: storage")
            logger.debug(f"Clean text length: {len(clean_text)}")
            logger.debug(f"Clean text preview: {clean_text[:200]}...")
            
            # Create chunks
            chunks = self.chunk_text(clean_text)
            logger.debug(f"Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'page_id': doc['id'],
                    'title': doc['title'],
                    'chunk_index': len(all_chunks) - 1,  # Global chunk position
                    'total_chunks': len(chunks),
                    'created': doc.get('created', ''),
                    'updated': doc.get('version', {}).get('when', '')
                })
        
        if not all_chunks:
            logger.warning("No valid chunks found")
            return
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = self.create_embeddings(all_chunks)
        
        # Save to FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        faiss.write_index(index, os.path.join(self.vector_db_path, 'faiss_index.bin'))
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(os.path.join(self.vector_db_path, 'metadata.csv'), index=False)
        
        # Save chunks
        chunks_df = pd.DataFrame({
            'chunk_id': range(len(all_chunks)),
            'text': all_chunks
        })
        chunks_df.to_csv(os.path.join(self.vector_db_path, 'chunks.csv'), index=False)
        
        logger.info(f"Vector database saved to {self.vector_db_path}")
        logger.info(f"Index contains {len(all_chunks)} chunks with {dimension}-dimensional embeddings")
    
    def prepare_rag_data(self) -> None:
        """Main method to prepare RAG data from Confluence"""
        logger.info("Starting RAG data preparation...")
        
        # Get all pages from Confluence
        pages = self.get_space_pages()
        if not pages:
            logger.error("No pages found in Confluence space")
            return
        
        # Get detailed content for each page
        documents = []
        for page in tqdm(pages, desc="Fetching page content"):
            page_id = page['id']
            content = self.get_page_content(page_id)
            if content:
                documents.append(content)
        
        # Build vector database
        self.build_vector_database(documents)
        
        logger.info("RAG data preparation completed successfully!")

    def cleanup_existing_chunks(self) -> None:
        """Clean up existing broken chunks and regenerate vector database"""
        import os
        import shutil
        
        logger.info("Cleaning up existing broken chunks...")
        
        # Remove existing vector database files
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
            logger.info(f"Removed existing vector database at {self.vector_db_path}")
        
        # Recreate directory
        os.makedirs(self.vector_db_path, exist_ok=True)
        logger.info("Recreated vector database directory")
    
    def regenerate_vector_database(self) -> None:
        """Regenerate the entire vector database with proper content processing"""
        logger.info("Starting vector database regeneration...")
        
        # Clean up existing broken chunks
        self.cleanup_existing_chunks()
        
        # Fetch fresh content from Confluence
        documents = self.get_space_pages()
        if not documents:
            logger.error("No documents found to process")
            return
        
        # Build new vector database
        self.build_vector_database(documents)
        
        logger.info("Vector database regeneration completed successfully!")

def main():
    """Main function to run the RAG preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Confluence RAG Preparation')
    parser.add_argument('--regenerate', action='store_true', 
                       help='Clean up existing chunks and regenerate vector database')
    args = parser.parse_args()
    
    try:
        rag_system = ConfluenceRAG()
        
        if args.regenerate:
            logger.info("Regenerating vector database...")
            rag_system.regenerate_vector_database()
        else:
            logger.info("Starting RAG data preparation...")
            rag_system.prepare_rag_data()
            
    except Exception as e:
        logger.error(f"Error during RAG preparation: {e}")
        raise

if __name__ == "__main__":
    main()
