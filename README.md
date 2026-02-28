# RAG Pipeline — BVZyme Semantic Search Engine

> Retrieval-Augmented Generation pipeline for searching technical enzyme datasheets (BVZyme product range).  
> Built with **sentence-transformers**, **FAISS**, and hybrid **cosine + BM25** scoring.

---

## Features

| Capability | Detail |
|---|---|
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim, cosine similarity) |
| **Vector index** | FAISS `IndexFlatIP` with L2-normalised embeddings |
| **Hybrid scoring** | Weighted cosine similarity + BM25 (percentile-normalised) + chunk quality |
| **Contextual Chunk Enrichment** | Each chunk is prepended with document title & opening context |
| **Adaptive chunking** | Small documents kept whole; larger ones split with overlap |
| **Query expansion** | FR→EN translation + synonym generation (max 3 for embedding, 8 for BM25) |
| **Dynamic weights** | Automatically adjusts cosine/BM25 balance per query type (product, numeric, long) |
| **MMR reranking** | Maximal Marginal Relevance with intra-document penalty differentiation |
| **Caching** | LRU query cache with config-aware fingerprinting |
| **Interfaces** | CLI, REST API (FastAPI), Web UI (Streamlit) |

---

## Project Structure

```
rag-pipeline/
├── config.py            # Centralised constants & weights
├── data_loader.py       # PDF / text extraction
├── text_cleaner.py      # Text normalisation
├── query_expander.py    # FR→EN expansion & synonyms
├── rag_pipeline.py      # Core pipeline (chunking, indexing, search)
├── baseline_pipeline.py # Naïve baseline for comparison
├── evaluate.py          # 30-question evaluation suite (P@1, MRR, NDCG@3)
├── search_cli.py        # Interactive CLI search
├── api.py               # FastAPI REST API
├── app.py               # Streamlit Web UI
├── test_pipeline.py     # Quick smoke tests
├── tests/               # Full pytest suite (34 tests)
│   ├── conftest.py
│   └── test_pipeline.py
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/FEDI-HASSINE/rag-pipeline.git
cd rag-pipeline
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### 2. Configure data folder

Copy `.env.example` to `.env` and set `RAG_DATA_FOLDER` to the directory containing your enzyme PDF datasheets:

```dotenv
RAG_DATA_FOLDER=C:/path/to/enzymes
```

### 3. Run

| Interface | Command |
|---|---|
| **CLI** | `python search_cli.py --reindex` |
| **API** | `uvicorn api:app --reload --port 8000` |
| **Web UI** | `streamlit run app.py` |
| **Evaluate** | `python evaluate.py --verbose` |
| **Tests** | `pytest tests/ -v` |

---

## Evaluation Results

Evaluated on **30 questions** (20 positive EN/FR + 7 specialised + 3 negative):

| Metric | Optimised Pipeline | Baseline |
|---|---|---|
| **P@1** | 70.0 % | 50.0 % |
| **P@3** | 83.3 % | 56.7 % |
| **MRR** | 0.756 | 0.533 |
| **NDCG@3** | 0.767 | 0.476 |

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "alpha-amylase dosage for bread", "top_k": 3}'

# Index statistics
curl http://localhost:8000/stats
```

Interactive Swagger docs available at **http://localhost:8000/docs**.

---

## Configuration

All tuneable parameters live in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBEDDING_DIM` | 384 | Embedding dimension |
| `TOP_K` | 3 | Number of results returned |
| `ALPHA_COSINE` | 0.80 | Default cosine weight |
| `ALPHA_BM25` | 0.10 | Default BM25 weight |
| `ALPHA_QUALITY` | 0.10 | Chunk quality weight |
| `MMR_LAMBDA` | 1.0 | MMR diversity (1.0 = disabled) |
| `QUERY_EXPANSIONS` | 3 | Max query expansions for embedding |

---

## Tech Stack

- **Python 3.10+**
- [sentence-transformers](https://www.sbert.net/) 2.7.0
- [faiss-cpu](https://github.com/facebookresearch/faiss) 1.8.0
- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- [Streamlit](https://streamlit.io/) 1.32+
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six) for PDF extraction
- [Rich](https://rich.readthedocs.io/) for terminal formatting
- numpy < 2 (faiss-cpu compatibility)

---

## License

MIT