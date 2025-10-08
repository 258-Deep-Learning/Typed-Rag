# Using Gemini 2.5 Flash with Typed-RAG

This project is now configured to use **Google Gemini 2.5 Flash** by default - it's faster and more cost-effective than GPT models!

## 🚀 Quick Setup

### 1. Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Get API Key" or "Create API Key"
3. Copy your API key

### 2. Set the Environment Variable

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

To make it permanent, add to your `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export GOOGLE_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Install the Gemini SDK

```bash
pip install google-generativeai
```

(Already included in `requirements.txt`)

## 📊 Supported Gemini Models

The scripts support any Gemini model. Common choices:

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| `gemini-2.0-flash-exp` | **Most tasks** (default) | ⚡️ Very Fast | 💰 Very Cheap |
| `gemini-1.5-flash` | Fast responses | ⚡️ Fast | 💰 Cheap |
| `gemini-1.5-pro` | Complex reasoning | 🐌 Slower | 💰💰 More expensive |

## 🎯 Using Different Models

Simply pass the `--model` flag:

```bash
# Use Gemini 2.0 Flash (default)
python3 typed_rag/scripts/run_rag_baseline.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/rag_baseline.jsonl \
  --model gemini-2.0-flash-exp

# Use Gemini 1.5 Pro for better quality
python3 typed_rag/scripts/run_rag_baseline.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/rag_baseline.jsonl \
  --model gemini-1.5-pro

# Or use OpenAI (if you have OPENAI_API_KEY set)
python3 typed_rag/scripts/run_rag_baseline.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/rag_baseline.jsonl \
  --model gpt-3.5-turbo
```

## 💡 Why Gemini?

- **Speed**: Gemini 2.0 Flash is extremely fast
- **Cost**: Much cheaper than GPT-3.5/4
- **Quality**: Excellent quality for most RAG tasks
- **Context**: Large context windows (up to 1M tokens for Gemini 1.5)
- **Free Tier**: Generous free quota for development

## 🔄 Switching Between Providers

The scripts auto-detect which API key is set:

```bash
# Use Gemini (checks GOOGLE_API_KEY first)
export GOOGLE_API_KEY="your-gemini-key"
python3 typed_rag/scripts/run_llm_only.py --in dev.jsonl --out results.jsonl

# Use OpenAI instead (checks OPENAI_API_KEY if Gemini not available)
unset GOOGLE_API_KEY
export OPENAI_API_KEY="your-openai-key"
python3 typed_rag/scripts/run_llm_only.py --in dev.jsonl --out results.jsonl --model gpt-3.5-turbo
```

## 📈 Rate Limits & Quotas

**Free Tier** (no payment required):
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per day

**Paid Tier**:
- Much higher limits
- Pay as you go pricing

Check current pricing: https://ai.google.dev/pricing

## 🐛 Troubleshooting

### "GOOGLE_API_KEY not set"

```bash
# Check if set
echo $GOOGLE_API_KEY

# If empty, set it
export GOOGLE_API_KEY="your-key"
```

### "API quota exceeded"

You've hit the rate limit. Wait a minute or upgrade to paid tier.

### "Model not found"

Make sure you're using a valid model name:
- ✅ `gemini-2.0-flash-exp`
- ✅ `gemini-1.5-flash`
- ✅ `gemini-1.5-pro`
- ❌ `gemini-2.5-flash` (not the correct name)

### Import Error

```bash
pip install google-generativeai
```

## 🎉 You're All Set!

Now run the pipeline:

```bash
export GOOGLE_API_KEY="your-key"
export PINECONE_API_KEY="your-pinecone-key"

./quickstart.sh my_documents
```

Happy RAG-ing with Gemini! 🚀

