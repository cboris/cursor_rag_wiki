# Confluence RAG System

A Retrieval-Augmented Generation (RAG) system that extracts content from Confluence wiki pages and provides intelligent question-answering capabilities using local LLMs.

Created completely with vibe coding using cursor pro plan.

Limitations: it got difficulties to retrieve using LLM/RAG approach poorly structured wiki pages with simple words. Get's much better on text rich wiki pages.


## Features

- **Confluence Integration**: Automatically extracts content from Confluence wiki spaces
- **Local Vector Database**: Uses FAISS for efficient similarity search
- **Local LLM Support**: Compatible with Ollama and other local LLM endpoints
- **Configurable**: Easy configuration via YAML file
- **Interactive Q&A**: Command-line interface for asking questions
- **Source Attribution**: Provides source page information for answers

## Prerequisites

- Python 3.8+
- Confluence  account with API access
- Local LLM (e.g., Ollama with Llama2)
- Internet connection for Confluence API access

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama** (for local LLM):
   ```bash
   # Install Ollama (macOS)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Llama2 model
   ollama pull llama2
   ```

## Configuration

1. **Edit `config.yaml`** with your Confluence credentials:
   ```yaml
   confluence:
     base_url: "https://your-domain.atlassian.net"
     username: "your-email@domain.com"
     api_token: "your-api-token"
     space_key: "SPACE"
   ```

2. **Get Confluence API Token**:
   - Go to [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)
   - Create a new API token
   - Use your email and the API token in the config

3. **Configure LLM endpoint** (default is Ollama):
   ```yaml
   llm:
     endpoint: "http://localhost:11434/api/generate"
     model: "llama2"
   ```

## Usage

### 1. Prepare RAG Data

First, extract content from Confluence and create the vector database:

```bash
python confluence_rag.py
```

This will:
- Connect to your Confluence space
- Extract all page content
- Create text chunks with overlapping
- Generate embeddings using sentence-transformers
- Store everything in a local FAISS vector database

### 2. Query the RAG System

#### Interactive Mode
```bash
python rag_query.py --interactive
```

#### Single Question
```bash
python rag_query.py --question "What is the deployment process?"
```

#### Custom Config File
```bash
python rag_query.py --config my_config.yaml --question "How do I set up authentication?"
```

## File Structure

```
cursor_rag_wiki/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── confluence_rag.py       # RAG preparation script
├── rag_query.py            # Query script
├── README.md               # This file
└── vector_db/              # Generated vector database (after running prep)
    ├── faiss_index.bin     # FAISS index file
    ├── metadata.csv        # Chunk metadata
    └── chunks.csv          # Text chunks
```

## Configuration Options

### Confluence Settings
- `base_url`: Your Confluence instance URL
- `username`: Your Confluence email
- `api_token`: Your API token
- `space_key`: The space key to extract content from

### LLM Settings
- `endpoint`: LLM API endpoint (default: Ollama)
- `model`: Model name to use
- `temperature`: Response randomness (0.0-1.0)
- `max_tokens`: Maximum response length

### Vector Database Settings
- `path`: Local path for storing the database
- `embedding_model`: Sentence transformer model for embeddings
- `chunk_size`: Size of text chunks
- `chunk_overlap`: Overlap between chunks

### RAG Settings
- `top_k`: Number of similar chunks to retrieve
- `similarity_threshold`: Minimum similarity score for chunks

## Troubleshooting

### Common Issues

1. **Confluence API Errors**:
   - Verify your API token is correct
   - Check that your account has access to the specified space
   - Ensure the space key is correct

2. **LLM Connection Issues**:
   - Make sure Ollama is running: `ollama serve`
   - Check the endpoint URL in config
   - Verify the model name matches what's available

3. **Memory Issues**:
   - Reduce chunk size in config
   - Use a smaller embedding model
   - Process smaller spaces at a time

### Performance Tips

- Use `sentence-transformers/all-MiniLM-L6-v2` for faster embeddings
- Adjust chunk size based on your content type
- Set appropriate similarity thresholds for your use case

## Security Notes

- **Never commit your `config.yaml`** with real credentials
- Store API tokens securely
- Consider using environment variables for sensitive data
- The vector database is stored locally and contains your content

## Customization

### Adding New LLM Providers

The system supports both Ollama and OpenAI-compatible APIs. To add a new provider:

1. Update the `query_llm` method in `rag_query.py`
2. Add your provider's request format
3. Update the response parsing logic

### Custom Embedding Models

You can use any sentence-transformers model:

```yaml
vector_db:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"  # Better quality, slower
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"  # Faster, good quality
```

## License

This project is provided as-is for educational and development purposes.


Last iteration of vibe coding and next possible directions:



## �� **The Real Problem:**

**poor semantic understanding**:


 The RAG system is fundamentally **biased toward long, descriptive content** and **struggles with short, specific page titles** like "Delegations" (from the wiki samples used).

The system works for queries like "How to set up xxxx upgrade" but fails for "find the Delegations page" because it can't semantically connect the query intent with the page structure.

This is a **fundamental limitation** of the current approach, not a bug we can patch! 🎯

Alternatives:
🚀 Alternative Approach 1: Multi-Vector Hybrid Search
Instead of just semantic search, combine multiple search strategies:
```
def hybrid_search(self, query: str):
    # 1. Page title search (exact matches)
    title_results = self.search_by_page_title(query)
    
    # 2. Named entity search (people, projects)
    entity_results = self.search_by_entities(query)
    
    # 3. Semantic search (fallback)
    semantic_results = self.semantic_search(query)
    
    # 4. Combine and rank intelligently
    return self.rank_and_merge_results(title_results, entity_results, semantic_results)
```
Pros: Leverages different strengths of each approach
Cons: More complex, needs entity extraction
🎯 Alternative Approach 2: Enhanced Chunking with Metadata
Modify the chunking strategy to include more context:
```
def create_enhanced_chunks(self, content):
    chunks = []
    for chunk in self.basic_chunks(content):
        enhanced_chunk = {
            'text': chunk['text'],
            'page_title': chunk['page_title'],
            'page_title_embedding': self.embed_page_title(chunk['page_title']),
            'entities': self.extract_entities(chunk['text']),  # Richard Chan, projects, etc.
            'chunk_type': self.classify_chunk(chunk['text'])   # table, list, paragraph
        }
        chunks.append(enhanced_chunk)
    return chunks
```
Pros: Richer context, better semantic understanding
Cons: More storage, slower indexing
🔍 Alternative Approach 3: Query Intent Classification
Route queries to appropriate search strategies:
semantic
```
def classify_query_intent(self, query):
    if "page" in query.lower() or "find" in query.lower():
        return "page_lookup"  # Use page title search
    elif any(person in query.lower() for person in ["richard", "boris", "tomasz"]):
        return "person_search"  # Use entity search
    elif "table" in query.lower() or "tasks" in query.lower():
        return "structured_data"  # Use table-aware search
    else:
        return "semantic_search"  # Default to semantic
```
Pros: Smart routing, better precision
Cons: Needs training data for intent classification

🧠 Alternative Approach 4: LLM-Powered Search
Use the LLM to understand and enhance search:
```
def llm_enhanced_search(self, query):
    # 1. LLM analyzes query intent
    intent = self.llm.analyze_query(query)
    
    # 2. LLM suggests search strategies
    search_strategies = self.llm.suggest_search_methods(intent)
    
    # 3. Execute multiple searches
    results = self.execute_search_strategies(search_strategies)
    
    # 4. LLM ranks and explains results
    return self.llm.rank_and_explain_results(query, results)
```

Pros: Most intelligent, understands context
Cons: Slower, more expensive, depends on LLM quality
🎯 Alternative Approach 5: Graph-Based Search
Build a knowledge graph from your content:
```
def build_knowledge_graph(self):
    # Extract entities and relationships
    entities = self.extract_entities_from_all_content()
    relationships = self.extract_relationships(entities)
    
    # Build graph
    graph = nx.Graph()
    graph.add_nodes_from(entities)
    graph.add_edges_from(relationships)
    
    return graph

def graph_search(self, query):
    # Find relevant entities
    query_entities = self.extract_entities(query)
    
    # Traverse graph to find related content
    related_content = self.traverse_graph(query_entities)
    
    return related_content
```

Pros: Captures relationships, great for entity queries
Cons: Complex to implement, needs relationship extraction
💡 My Recommendation:
For your specific use case (Confluence with tables, people, and short page titles), I'd recommend Approach 2 + 3:
1. Enhanced chunking with page titles and entities
2. Query intent classification to route to appropriate search
3. Keep semantic search as fallback
This gives you:
✅ Reliable page title matching for "find Delegations page"
✅ Entity-based search for "Richard Chan tasks"
✅ Table-aware chunking for delegation data
✅ Semantic fallback for complex queries
Would you like me to implement one of these approaches? Which one interests you most? 🚀
