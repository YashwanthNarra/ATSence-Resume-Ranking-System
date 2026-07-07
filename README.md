# AI-Based Resume Intelligence System

An AI-powered Resume Intelligence System that analyzes resumes against job descriptions and provides ATS-style scoring, candidate ranking, semantic skill matching, and personalized improvement suggestions. Unlike traditional keyword-based ATS systems, this project leverages Large Language Models (LLMs) and NLP techniques to understand the semantic meaning of resumes and job descriptions, enabling more accurate candidate evaluation.

---

## Features

- 📄 Resume parsing from PDF using Gemini Vision
- 🧠 AI-based extraction of structured resume information
- 💼 Job Description (JD) requirement extraction
- 🔍 Semantic skill matching using Sentence Transformers
- 📊 Weighted ATS score calculation
- 📈 Detailed score breakdown
- ⭐ Candidate ranking
- 📝 AI-generated gap analysis
- 💡 Personalized resume improvement suggestions
- ⚡ Built with streamlit

---

## Tech Stack

### Backend
- Python
- Streamlit

### AI & NLP
- Google Gemini 2.0 Flash
- Sentence Transformers
- spaCy

### Machine Learning
- scikit-learn
- NumPy
- pandas

### Other Libraries
- Regular Expressions (Regex)

---

## System Architecture

Resume PDF
↓
Gemini Vision
↓
Structured Resume JSON
↓
Job Description Parsing
↓
Semantic Skill Matching
↓
ATS Score Calculation
↓
Gap Analysis
↓
Candidate Ranking & Suggestions

---

## Project Structure

```
Resume_Intelligence_System/
│
├── app.py
├── extractor.py
├── scorer.py
├── requirements.txt
├── README.md
└── ...
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YashwanthNarra/AI-based resume ranking system.git
cd AI-based resume ranking system
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file

```text
GEMINI_API_KEY=your_api_key
```

---

## Running the Project

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

to access the FastAPI Swagger UI.

---

## Workflow

1. Upload Resume PDF.
2. Provide Job Description.
3. Gemini extracts structured resume information.
4. JD is converted into structured requirements.
5. Skills are matched using semantic similarity.
6. ATS score is calculated using weighted scoring.
7. Gap analysis identifies missing skills.
8. Personalized suggestions are generated.
9. Candidates are ranked based on overall compatibility.

---

## Future Improvements

- Multi-resume ranking
- Experience relevance scoring
- Education matching
- Project relevance analysis
- Resume keyword highlighting
- Better PDF layout handling
- Database integration
- Authentication and user management
- Docker deployment
- Cloud deployment

---

## Key Highlights

- AI-powered resume parsing using Gemini Vision
- Semantic matching instead of simple keyword matching
- Production-ready REST API using FastAPI
- Structured ATS scoring methodology
- Personalized AI feedback for resume improvement

---
