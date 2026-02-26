# 🎮 Game Composer Agent v1

> **AI-powered MCQ game generator** — Reads 4 PDFs, generates 20 challenging questions (5 per chapter), separates public/private data, and produces blockchain-compatible anchor strings.

Built with **FastAPI + Groq (Llama 3.3 70B) + PyMuPDF + Pydantic v2**.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │          POST /generate-game                │
                        │          { "difficulty_target": 700 }       │
                        └─────────────┬───────────────────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   For each chapter 1–4  │──────────────────────┐
                         └────────────┬────────────┘                      │
                                      │                                   │
                    ┌─────────────────▼──────────────────┐               │
                    │  1. PDF Loader (security + extract) │               │
                    └─────────────────┬──────────────────┘               │
                                      │                                   │
                    ┌─────────────────▼──────────────────┐               │
                    │  2. Chunker (semantic sentences)    │               │
                    └─────────────────┬──────────────────┘               │
                                      │                                   │
                    ┌─────────────────▼──────────────────┐               │
                    │  3. LLM Generator (Groq/Llama 3.3) │               │
                    │     5 questions per chapter         │               │
                    │     + option shuffle (A/B/C/D fix)  │               │
                    └─────────────────┬──────────────────┘               │
                                      │                                   │
                    ┌─────────────────▼──────────────────┐               │
                    │  4. Validator (10 integrity rules)  │               │
                    └─────────────────┬──────────────────┘               │
                                      │                                   │
                                      └───────────── × 4 chapters ───────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │  5. Formatter (merge 4 → 20 Qs)    │
                    │     public/private separation       │
                    │     blockchain anchor strings       │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │  GameResponse JSON (20 MCQs)       │
                    │  • public_puzzles  → frontend safe  │
                    │  • private_answer_key → backend only│
                    └────────────────────────────────────┘
```

## Features

- **4 chapters from 4 different PDFs** — each chapter = one knowledge domain
- **5 questions per chapter = 20 total** — medium (2) + hard (3) difficulty
- **Non-deterministic** — fresh, different questions every call (`temperature=0.7`)
- **Option shuffling** — correct answers evenly distributed across A/B/C/D (fixes LLM bias)
- **Public/private separation** — public puzzles contain NO answers; private keys stay on backend
- **Blockchain anchors** — `ch1_q01|B` format for on-chain verification
- **Groq rate-limit handling** — auto-retry + inter-chapter delays
- **Token budget management** — selects chunks within Groq free-tier 12K TPM limit
- **PDF security** — path traversal prevention, size limits, extension validation
- **24 unit tests** — full pipeline coverage

## Chapters

| Chapter | Topic | Source PDF |
|---------|-------|-----------|
| 1 | AI, Machine Learning & Deep Learning | `Artificial Intelligence, Machine Learning, and Deep Learning.pdf` |
| 2 | NLP & Large Language Models | `room2_nlp_llms.pdf` |
| 3 | Agentic Cybersecurity | `room3_agentic_cybersec (1).pdf` |
| 4 | Bitcoin Vocabulary | `bitcoin_vocabulary.pdf` |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/OussemaAissaoui1/GameComposerAgent.git
cd GameComposerAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 3. Place PDFs

Place your 4 chapter PDFs in the configured directory (default: parent directory).
Update `CHAPTER_PDF_MAP` in `app/config.py` if paths differ.

### 4. Run Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Generate a Game

```bash
curl -X POST http://localhost:8000/generate-game \
  -H "Content-Type: application/json" \
  -d '{"difficulty_target": 700}'
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

## API Reference

### `POST /generate-game`

Generate 20 MCQs across 4 chapters.

**Request:**
```json
{
  "difficulty_target": 700
}
```
> `difficulty_target` is optional (default: 700, range: 400–1000).

**Response:**
```json
{
  "status": "success",
  "game": {
    "meta": {
      "total_questions": 20,
      "questions_per_chapter": 5,
      "chapters": ["1", "2", "3", "4"],
      "chapter_titles": {
        "1": "Artificial Intelligence, Machine Learning, and Deep Learning",
        "2": "Natural Language Processing and Large Language Models",
        "3": "Agentic Cybersecurity",
        "4": "Bitcoin Vocabulary"
      },
      "difficulty_target": 700,
      "difficulty_distribution_per_chapter": {"medium": 2, "hard": 3},
      "model_used": "llama-3.3-70b-versatile",
      "temperature": 0.7,
      "version": "1.0.0"
    },
    "public_puzzles": [
      {
        "puzzle_id": "ch1_q01",
        "chapter_id": "1",
        "chapter_title": "Artificial Intelligence, Machine Learning, and Deep Learning",
        "question": "How do Random Forests operate, and what is the underlying assumption?",
        "options": [
          {"option_id": "A", "text": "By using a single decision tree..."},
          {"option_id": "B", "text": "By using multiple decision trees..."},
          {"option_id": "C", "text": "By using a neural network..."},
          {"option_id": "D", "text": "By using a support vector machine..."}
        ],
        "difficulty": "hard",
        "difficulty_rating": 750,
        "min_solve_time_seconds": 60,
        "source_chunk_index": 2
      }
    ],
    "private_answer_key": [
      {
        "puzzle_id": "ch1_q01",
        "chapter_id": "1",
        "correct_option_id": "B",
        "anchor_string": "ch1_q01|B",
        "explanation": "According to the text, Random Forests operate by..."
      }
    ]
  }
}
```

### `GET /`

Service info and available endpoints.

### `GET /health`

Health check — returns `{"status": "healthy"}`.

### `GET /docs`

Interactive Swagger UI (auto-generated by FastAPI).

## React Integration

The agent is designed as a backend service for a React app:

```
React Frontend          Your Backend (Next.js/Express)         Game Composer Agent
     │                              │                                   │
     │── POST /api/game/new ───────▶│                                   │
     │                              │── POST /generate-game ───────────▶│
     │                              │◀── { public_puzzles, private } ───│
     │                              │                                   │
     │                              │ Store full game (with answers)     │
     │◀── { game_id, puzzles } ─────│ Send ONLY public_puzzles          │
     │                              │                                   │
     │ User plays quiz...           │                                   │
     │                              │                                   │
     │── POST /api/game/submit ────▶│                                   │
     │   { game_id, answers }       │ Grade against stored private keys │
     │◀── { score, results } ───────│                                   │
```

**Key rule:** `public_puzzles` → safe for frontend. `private_answer_key` → **NEVER** send to frontend.

## Project Structure

```
game_maker/
├── app/
│   ├── __init__.py
│   ├── config.py              # Settings, chapter-PDF mapping
│   ├── main.py                # FastAPI app factory
│   ├── routes.py              # POST /generate-game endpoint
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # All Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── pdf_loader.py      # Secure PDF text extraction
│       ├── chunker.py         # Semantic sentence-based chunking
│       ├── llm_generator.py   # Groq LLM orchestration + option shuffle
│       ├── validator.py       # 10 integrity validation rules
│       └── formatter.py       # Public/private separation + anchors
├── tests/
│   ├── __init__.py
│   └── test_generation.py     # 24 unit tests
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | LLM model identifier |
| `LLM_TEMPERATURE` | `0.7` | Higher = more varied questions per call |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `PDF_ALLOWED_DIRECTORY` | `/home/oussema/3` | Allowed directory for PDF files |
| `PDF_MAX_SIZE_BYTES` | `52428800` | Max PDF file size (50MB) |

## How It Works

1. **PDF Loading** — Each chapter's PDF is securely loaded and full text extracted via PyMuPDF
2. **Semantic Chunking** — Text split into sentence-based chunks with overlap for context
3. **Chunk Budget** — Random subset of chunks selected within Groq's 12K TPM token limit
4. **LLM Generation** — Groq Llama 3.3 70B generates 5 MCQs per chapter (temp=0.7 for variety)
5. **Option Shuffling** — Correct answers randomly redistributed across A/B/C/D (max 2 per letter per chapter)
6. **Validation** — 10 rules checked: question count, options, difficulty distribution, duplicates, rating ranges
7. **Formatting** — 4 chapters merged into 20-question payload with public/private separation
8. **Anchor Strings** — `puzzle_id|correct_option_id` format for blockchain verification

## License

MIT
