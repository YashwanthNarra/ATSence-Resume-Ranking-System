# Resume Intelligence System

Upload resumes + a job description → get ATS scores, rankings, and gap analysis.

---

## Project Structure

```
resume_system/
├── main.py           ← FastAPI backend (all API endpoints)
├── extractor.py      ← Gemini API calls (parse resume, parse JD, gap analysis)
├── scorer.py         ← Scoring logic (skill match, project rank, ATS score)
├── index.html        ← Frontend (open in browser, no extra setup needed)
└── requirements.txt
```

---

## How to Run

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Start the backend**
```bash
uvicorn main:app --reload
```

**Step 3 — Open the frontend**

Just open `index.html` in your browser (double-click it or drag it into Chrome).

---

## API Endpoints

| Method | Endpoint          | What it does                                      |
|--------|-------------------|---------------------------------------------------|
| GET    | `/health`         | Check if the server is running                    |
| POST   | `/analyse`        | Upload resumes + JD → returns ranked list         |
| GET    | `/rankings`       | Get the rankings from the last /analyse call      |
| GET    | `/resume/{id}`    | Full details for one resume (use id from rankings)|

### POST /analyse — Form fields

| Field     | Type         | Description                         |
|-----------|--------------|-------------------------------------|
| `resumes` | PDF files    | One or more resume PDFs             |
| `jd_text` | string       | The job description as plain text   |
| `api_key` | string       | Your Gemini API key                 |

### GET /resume/{id}

Use the `id` field from the `/analyse` or `/rankings` response.

Example: `/resume/0` → details for the #1 ranked candidate.

---

## Scoring Weights

| Component  | Weight |
|------------|--------|
| Skills     | 50%    |
| Projects   | 25%    |
| Experience | 15%    |
| Education  | 10%    |

---

## How It Works

1. PDF is sent directly to **Gemini Vision** — handles any layout including multi-column
2. JD is parsed by Gemini to extract required skills, experience, education
3. Skills are matched using **sentence embeddings** (cosine similarity, not exact keywords)
4. Projects are **ranked by relevance** to the JD
5. All components are combined into a **weighted ATS score**
6. Gemini generates a **gap analysis** with actionable suggestions

---

## Tech Stack

- FastAPI + Uvicorn (backend)
- Google Gemini 2.0 Flash (resume parsing, JD parsing, gap analysis)
- intfloat/e5-small-v2 via sentence-transformers (semantic skill matching)
- scikit-learn cosine similarity
- Vanilla HTML/CSS/JS (frontend — no framework needed)
