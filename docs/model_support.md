# Model Support in Typed-RAG

## Available Models

The Typed-RAG system now supports **4 models** across all 3 system variants (LLM-Only, RAG Baseline, Typed-RAG):

### Google Gemini Models
1. **gemini-2.5-flash**
   - Latest flagship Gemini model
   - Best overall performance (MRR: 0.333 for LLM-only)
   - Requires: `GEMINI_API_KEY` environment variable

2. **gemini-2.0-flash-lite**
   - Lightweight, faster Gemini variant
   - Good performance (MRR: 0.332 for LLM-only)
   - Lower cost and latency
   - Requires: `GEMINI_API_KEY` environment variable

### Groq-Hosted Models
3. **groq/llama-3.3-70b-versatile**
   - Meta's Llama 3.3 70B model via Groq
   - Strong reasoning capabilities
   - Best Typed-RAG performance (MRR: 0.288)
   - Requires: `GROQ_API_KEY` environment variable

4. **groq/llama-3.1-8b-instant**
   - Compact Llama 3.1 8B model via Groq
   - Fast inference, lower resource usage
   - Competitive performance (MRR: 0.313 for Typed-RAG)
   - Requires: `GROQ_API_KEY` environment variable

## Evaluation Results Summary

All 12 system configurations have been fully evaluated on the wiki_nfqa dev100 dataset (97 questions):

| Model | LLM-Only MRR | RAG Baseline MRR | Typed-RAG MRR |
|-------|--------------|------------------|---------------|
| Gemini 2.5 Flash | 0.333 | 0.203 | 0.302 |
| Gemini 2.0 Flash-Lite | 0.332 | 0.175 | 0.237 |
| Llama 3.3 70B | 0.278 | 0.191 | 0.288 |
| Llama 3.1 8B | 0.277 | 0.248 | 0.313 |

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# For Gemini models
GEMINI_API_KEY=your_gemini_api_key_here

# For Groq models (Llama)
GROQ_API_KEY=your_groq_api_key_here
```

### Streamlit UI

The interactive Streamlit UI ([app.py](../app.py)) now includes all 4 models in the model selection dropdown:

```bash
streamlit run app.py
```

Access at: http://localhost:8501

### Command-Line Usage

Run any system with any model:

```bash
# LLM-Only
python scripts/run_llm_only.py --model gemini-2.0-flash-lite --input data/wiki_nfqa/dev6.jsonl

# RAG Baseline
python scripts/run_rag_baseline.py --model groq/llama-3.1-8b-instant --input data/wiki_nfqa/dev6.jsonl

# Typed-RAG
python scripts/run_typed_rag.py --model gemini-2.5-flash --input data/wiki_nfqa/dev6.jsonl
```

## API Key Setup

### Gemini API Key
1. Visit: https://aistudio.google.com/apikey
2. Create a new API key
3. Add to `.env`: `GEMINI_API_KEY=your_key`

### Groq API Key
1. Visit: https://console.groq.com/keys
2. Create a new API key
3. Add to `.env`: `GROQ_API_KEY=your_key`

## Performance Characteristics

### Speed vs Quality Tradeoffs

**Fastest (Low Cost)**:
- `gemini-2.0-flash-lite` - Best for prototyping
- `groq/llama-3.1-8b-instant` - Fast Groq inference

**Balanced**:
- `gemini-2.5-flash` - Good speed/quality balance
- `groq/llama-3.3-70b-versatile` - Powerful reasoning

### Recommended Use Cases

- **Production**: `gemini-2.5-flash` or `groq/llama-3.3-70b-versatile`
- **Development/Testing**: `gemini-2.0-flash-lite` or `groq/llama-3.1-8b-instant`
- **Cost-Sensitive**: Groq models (free tier available)
- **Best Typed-RAG**: `groq/llama-3.1-8b-instant` (MRR: 0.313)

## Model-Specific Notes

### Gemini Models
- Excellent at Evidence and Comparison questions
- Native support for structured outputs
- Best parametric knowledge (strong LLM-only performance)

### Llama Models (via Groq)
- Better with Typed-RAG architecture (benefit more from retrieval)
- Strong reasoning capabilities
- Groq provides ultra-fast inference (~300 tokens/sec)

## System Requirements

All models run via API calls - no local GPU required!

**Memory**: Minimal (embeddings cached)
**Disk**: ~500MB (FAISS index)
**Network**: Required for API calls

## Future Model Support

The system is designed to easily add new models:
1. Add model name to [app.py](../app.py) dropdown (line 349-357)
2. Ensure API key environment variable is set
3. Run evaluation scripts with `--model` flag

Compatible with any Langchain-supported chat model!
