
# Professor Aurelius Actions API (Render-ready)

This is a FastAPI backend implementing eight endpoints used by your GPT Actions:
- /schedule_study
- /generate_flashcards
- /tenta_stats
- /generate_exam
- /export_proof
- /weekly_report
- /gap_analysis
- /grade_quiz

## Deploy on Render (Free)

1. Create a new GitHub repo and push these files (`server.py`, `requirements.txt`).
2. On https://render.com: **New → Web Service → Connect Repo**.
3. Environment: **Python 3.11+**.
4. Build command: `pip install -r requirements.txt`
5. Start command: `python server.py`  (Render will set `PORT`; server uses it automatically)
6. When deployed, note your base URL, e.g. `https://aurelius-actions.onrender.com`.

## Use in GPT Builder Actions

- In "Add Action", paste your OpenAPI schema (the JSON files I provided) and set the server URL to your Render URL.
- Alternatively, you can let GPT autodiscover from `https://YOUR_URL/openapi.json` (FastAPI serves it automatically).

## Files endpoint
Generated files (CSV, PNG, PDFs) are served from `/files/{filename}`.

