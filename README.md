# ⛳ Golf Analytics Platform

A smart dashboard for golf course admins — with a **rate calendar**, **analytics charts**, and an **AI chatbot** that answers questions about your data.

---

## 🤔 What Does It Do?

| Feature | Description |
|---|---|
| 🗓️ Rate Calendar | See daily tee time prices vs market rates on a colour-coded calendar |
| 📊 Analytics | View price charts by course, hour, and booking channel |
| 🤖 AI Chatbot | Ask questions in plain English — get structured business insights |

---

## 🏗️ How It Works

```
You type a question
       ↓
NLP cleans & expands your query
       ↓
BM25 + FAISS finds the most relevant data chunks
       ↓
Gemini / OpenAI turns them into a clear answer
       ↓
📊 Insight  📈 Analysis  💡 Recommendation
```

---

## � Project Structure

```
golf-analytics/
│
├── project.py          ← Run this (Streamlit dashboard)
│
├── backend/
│   └── app.py          ← FastAPI server (handles /chat, /health) on port 8001
│
├── rag/
│   ├── nlp.py          ← NLP: stopwords, synonyms, query expansion
│   ├── ingest.py       ← Reads CSVs → builds BM25 + FAISS index
│   ├── retriever.py    ← Finds relevant data for a question
│   └── llm.py          ← Sends data to Gemini/OpenAI → returns answer
│
├── chatbot/
│   └── widget.html     ← Floating chat button (bottom-right)
│
├── tests/              ← 28 automated tests
├── Availability.csv    ← Tee time data
├── Market Rates.csv    ← Pricing & occupancy data
├── .env                ← Your API keys (never share this)
└── requirements.txt    ← Python packages
```

---

## 🚀 Setup (Step by Step)

### 1. Create a virtual environment

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 2. Install packages

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Add your API key

Copy `.env.example` to `.env` and fill in your Gemini key:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

> Get a free key at → https://aistudio.google.com/app/apikey

### 4. Build the search index

```bash
.venv\Scripts\python.exe -m rag.ingest
```

> Only needed once. Run with `--force` to rebuild.

---

## ▶️ Running the App

Open **two terminals**:

**Terminal 1 — Backend**
```bash
.venv\Scripts\python.exe -m uvicorn backend.app:app --port 8001 --reload
```

**Terminal 2 — Dashboard**
```bash
.venv\Scripts\python.exe -m streamlit run project.py --server.port 8501
```

Then open → **http://localhost:8501**

---

## 🤖 Using the Chatbot

Click the **⛳ green button** at the bottom-right of the dashboard.

**Example questions you can ask:**

- `"average price for stonegate?"`
- `"which course has the highest occupancy?"`
- `"available tee times on GolfNow?"`
- `"compare prices across all courses"`
- `"show all course names"`

**Every answer looks like this:**

```
📊 Insight:      The key result in plain language
📈 Analysis:     What it means for your business
💡 Recommendation:  What action you can take
⚠️ Note:         Only shown when there's a warning
```

---

## � Environment Variables

| Variable | Required | What it's for |
|---|---|---|
| `LLM_PROVIDER` | ✅ | `gemini` or `openai` |
| `GEMINI_API_KEY` | ✅ | Your Gemini API key |
| `OPENAI_API_KEY` | ❌ | Only if using OpenAI |
| `GEMINI_MODEL` | ❌ | Default: `gemini-1.5-flash` |

---

## 🧪 Run Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

28 tests covering response formatting, fallback paths, and formatting rules.

---

## 🛠️ Common Issues

| Problem | Fix |
|---|---|
| `pip install` fails | Use `.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Chatbot not responding | Make sure FastAPI is running on port 8001 |
| `GEMINI_API_KEY not set` | Add your key to `.env` |
| FAISS index not found | Run `python -m rag.ingest` first |
| Dashboard not loading | Check internet — data loads from Google Drive |

---

## 📦 Tech Stack

| What | Technology |
|---|---|
| Dashboard | Streamlit + Plotly |
| Backend API | FastAPI + Uvicorn |
| NLP | Custom tokeniser, BM25, synonym expansion |
| Vector Search | FAISS (CPU) |
| LLM | Google Gemini / OpenAI GPT |
| Language | Python 3.12 |

---

*Built for golf course administrators* ⛳
