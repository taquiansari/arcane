# Arcane — Agentic Research Intelligence Platform

> Multi-agent research system with RAG pipeline, LangGraph orchestration, and CrewAI task delegation.

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for Redis)
- Cohere API key ([get one free](https://dashboard.cohere.com/api-keys))

### Setup

```bash
# 1. Clone and enter the project
cd arcane

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
copy .env.example .env
# Edit .env and add your COHERE_API_KEY

# 5. Start Redis
docker compose up -d

# 6. Run a research query
arcane research "What are the latest advances in protein folding?"

# 7. Or start the API server
arcane serve
```

## Architecture

Arcane uses a layered architecture:

- **LangGraph** — Stateful graph orchestration with conditional routing and checkpointing
- **CrewAI** — Role-based multi-agent collaboration (Planner, Researcher, Critic, Synthesizer)
- **Redis** — Vector storage, semantic caching, session state, and conversation memory
- **Cohere** — LLM generation, embeddings, and reranking
- **DuckDuckGo** — Free web search (no API key required)
- **FastAPI** — REST API + WebSocket streaming

## License

MIT
