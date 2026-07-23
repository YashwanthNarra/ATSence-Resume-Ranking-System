"""
main.py
FastAPI backend for the Resume Intelligence System.

Endpoints:
  GET  /               → serves the frontend HTML
  GET  /health         → confirms server is running
  POST /analyse        → upload resumes + JD → returns ranked list
  GET  /resume/{rank}  → full detail for one candidate by rank

Run:
  uvicorn main:app --reload
  Open http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from extractor import get_gemini_model, extract_resume_data, extract_jd_data, generate_gap_analysis
from scorer import analyse_resume

app = FastAPI(title="Resume Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store results in memory between /analyse and /resume/{rank} calls
stored_results: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/analyse")
async def analyse(
    resumes: list[UploadFile] = File(...),
    jd_text: str              = Form(...),
    api_key: str              = Form(...),
):
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume PDF.")

    model   = get_gemini_model(api_key)
    jd_data = extract_jd_data(jd_text, model)
    results = []

    for resume_file in resumes:
        pdf_bytes   = await resume_file.read()
        resume_data = extract_resume_data(pdf_bytes, model)

        # Skip if Gemini returned nothing useful
        if not resume_data.get("name") and not resume_data.get("skills"):
            continue

        gap    = generate_gap_analysis(resume_data, jd_text, model)
        result = analyse_resume(resume_data, jd_data, jd_text, gap)
        result["filename"] = resume_file.filename
        results.append(result)

    # Sort highest score first, then add rank numbers
    results.sort(key=lambda r: r["ats_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    global stored_results
    stored_results = results

    # Return just the summary for the ranking table
    summary = [
        {
            "rank":             r["rank"],
            "name":             r["name"] or r["filename"],
            "filename":         r["filename"],
            "ats_score":        r["ats_score"],
            "skill_score":      r["skill_score"],
            "project_score":    r["project_score"],
            "experience_score": r["experience_score"],
            "education_score":  r["education_score"],
        }
        for r in results
    ]

    return {
        "total":        len(results),
        "role_summary": jd_data.get("role_summary", ""),
        "rankings":     summary,
    }


@app.get("/resume/{rank}")
def get_resume_detail(rank: int):
    if not stored_results:
        raise HTTPException(status_code=404, detail="No results yet. Run /analyse first.")

    match = next((r for r in stored_results if r["rank"] == rank), None)

    if match is None:
        raise HTTPException(status_code=404, detail=f"No candidate at rank {rank}.")

    return match
