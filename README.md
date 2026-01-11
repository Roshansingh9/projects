# Kharagpur Data Science Hackathon 2026 - Track A Submission

## Overview

This system determines whether a hypothetical character backstory is **globally consistent** with a complete novel using multi-agent deliberation powered by **Groq's free API**.

### Why Groq?
- **Free access** to multiple high-performance models
- **Fast inference**: Groq's LPU™ technology provides extremely fast responses
- **Multi-model strategy**: Use different models for different tasks
  - `llama-3.3-70b-versatile`: Complex reasoning (prosecutor, judge)
  - `llama-3.1-8b-instant`: Fast analysis (defense)

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. **Get Groq API Key**:
   - Visit https://console.groq.com
   - Create free account
   - Generate API key

2. **Set API Key** in `config.yaml`:
```yaml
   llm:
     api_key: "YOUR_GROQ_API_KEY_HERE"
```

3. **Adjust models** (optional):
```yaml
   models:
     claim_extraction: "llama-3.3-70b-versatile"
     prosecutor: "llama-3.3-70b-versatile"
     defense: "llama-3.1-8b-instant"
     judge: "llama-3.3-70b-versatile"
```

## Available Groq Models (Free Tier)

| Model | Speed | Use Case |
|-------|-------|----------|
| llama-3.3-70b-versatile | Medium | Complex reasoning, final judgments |
| llama-3.1-8b-instant | Very Fast | Quick analysis, defense arguments |
| mixtral-8x7b-32768 | Fast | Long context (32k tokens) |
| llama-3.1-70b-versatile | Medium | Alternative to 3.3 |

## Execution
```bash
python run_inference.py
```

**Runtime:** ~3-5 minutes per sample (Groq is very fast!)

## Key Features

✅ **Free API**: Groq's generous free tier  
✅ **Multi-Model**: Task-specific model selection  
✅ **Fast Inference**: Groq's LPU acceleration  
✅ **Track A Compliant**: Pure NLP/GenAI  
✅ **Reproducible**: Single command execution  

## Performance Notes

- Groq's speed allows processing 100+ samples in reasonable time
- Rate limits (30 req/min) are handled automatically
- Multi-model strategy optimizes cost vs. quality trade-off