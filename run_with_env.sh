#!/bin/bash
# Helper script that loads .env and runs the pipeline

set -e

echo "🔑 Loading environment variables..."

# Try to find .env file
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "typed_rag/.env" ]; then
    ENV_FILE="typed_rag/.env"
elif [ -f "typed_rag/config.env.example" ]; then
    echo "⚠️  No .env file found!"
    echo "Please create one from the example:"
    echo "  cp typed_rag/config.env.example .env"
    echo "  # Edit .env and add your API keys"
    exit 1
else
    echo "❌ No .env file found"
    exit 1
fi

# Load the .env file
echo "✓ Loading from: $ENV_FILE"
export $(grep -v '^#' "$ENV_FILE" | xargs)

# Verify keys are set
if [ -z "$PINECONE_API_KEY" ]; then
    echo "❌ PINECONE_API_KEY not set in $ENV_FILE"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Neither GOOGLE_API_KEY nor OPENAI_API_KEY set in $ENV_FILE"
    echo "LLM baselines will use dummy responses"
fi

echo "✓ PINECONE_API_KEY loaded"
[ ! -z "$GOOGLE_API_KEY" ] && echo "✓ GOOGLE_API_KEY loaded"
[ ! -z "$OPENAI_API_KEY" ] && echo "✓ OPENAI_API_KEY loaded"
echo ""

# Run the quickstart
./quickstart.sh "$@"

