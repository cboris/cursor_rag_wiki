#!/bin/bash

# Confluence RAG System Setup Script

echo "🚀 Setting up Confluence RAG System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing Ollama..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "❌ Unsupported OS. Please install Ollama manually from https://ollama.ai"
        exit 1
    fi
else
    echo "✅ Ollama found: $(ollama --version)"
fi

# Create config file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "📝 Creating config.yaml from example..."
    cp example_config.yaml config.yaml
    echo "⚠️  Please edit config.yaml with your Confluence credentials!"
else
    echo "✅ config.yaml already exists"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your Confluence credentials"
echo "2. Start Ollama: ollama serve"
echo "3. Pull Llama2 model: ollama pull llama2"
echo "4. Run RAG preparation: python3 confluence_rag.py"
echo "5. Query the system: python3 rag_query.py --interactive"
echo ""
echo "For more information, see README.md"
