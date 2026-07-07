"""
extractor.py
All Gemini API calls live here:
  - read_pdf_bytes()          → raw bytes from any PDF source
  - extract_resume_data()     → Gemini vision reads PDF directly → structured JSON
  - extract_jd_data()         → required skills + experience from JD text
  - generate_gap_analysis()   → actionable feedback comparing resume vs JD

Why Gemini vision instead of pdfplumber:
  pdfplumber reads raw PDF text streams left-to-right, which completely
  breaks multi-column, designed resumes (two columns get interleaved,
  styled skill tags are dropped, icon-adjacent text is lost).
  Gemini 2.0 Flash accepts the raw PDF bytes as a file part and
  understands the visual layout — columns, styled sections, tables —
  exactly as a human would read it. No text extraction step needed.
"""

import json
import re
import io
import google.generativeai as genai


# ── Gemini client ─────────────────────────────────────────────────────────────

def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


# ── PDF bytes helper ──────────────────────────────────────────────────────────

def read_pdf_bytes(pdf_source) -> bytes:
    """
    Returns raw PDF bytes from either:
      - a file path string
      - any file-like object (Streamlit UploadedFile, BytesIO, etc.)
    """
    if hasattr(pdf_source, "read"):
        raw = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)   # allow re-reads if needed
        return raw
    with open(pdf_source, "rb") as f:
        return f.read()


# ── JSON response parser ──────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """
    Strips markdown code fences (```json ... ```) then parses JSON.
    Raises ValueError on parse failure so callers can fall back gracefully.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON: {e}\n\nRaw response:\n{raw}"
        ) from e


# ── Resume extraction via Gemini vision ───────────────────────────────────────

RESUME_EXTRACTION_PROMPT = """
You are a resume parser. The attached file is a resume PDF.
Read it visually — respect the layout, columns, and styled sections.

Return ONLY a valid JSON object — no explanation, no markdown fences — with this exact schema:

{
  "name": "Candidate full name or empty string",
  "skills": ["skill1", "skill2"],
  "projects": [
    {
      "title": "Project title",
      "description": "What it does and how",
      "technologies": ["tech1", "tech2"]
    }
  ],
  "experience": [
    {
      "role": "Job title",
      "company": "Company name",
      "duration_text": "Jan 2022 – Present",
      "duration_years": 1.5,
      "description": "Key responsibilities and achievements"
    }
  ],
  "education": {
    "degree": "e.g. B.Tech / BFA / Bachelor's",
    "branch": "e.g. Computer Science / Photography",
    "institution": "University or college name",
    "year": "Graduation year or empty string"
  }
}

Rules:
- skills: collect every technical skill, tool, software, language, framework, or methodology
  visible anywhere on the resume — including sidebar skill bars, certificate names,
  portfolio tools, and inline mentions inside job descriptions
- duration_years: calculate from duration_text as a float; use 0.0 if not determinable
- projects: include portfolio pieces, freelance work, and personal projects —
  not just sections explicitly labelled "Projects"
- If a section is absent, use an empty list / empty object / empty string — never omit a key
- Return nothing except the JSON object
"""


def extract_resume_data(pdf_source, model) -> dict:
    """
    Sends the PDF directly to Gemini as a vision input.
    Works on any resume layout: single-column, multi-column, designed templates.

    Args:
        pdf_source: file path (str) OR file-like object (Streamlit UploadedFile / BytesIO)
        model:      Gemini GenerativeModel instance

    Returns:
        Structured resume dict.
    """
    pdf_bytes = read_pdf_bytes(pdf_source)

    # Gemini inline data part — send the PDF bytes directly
    pdf_part = {
        "inline_data": {
            "mime_type": "application/pdf",
            "data": pdf_bytes,           # SDK accepts raw bytes here
        }
    }

    try:
        response = model.generate_content([RESUME_EXTRACTION_PROMPT, pdf_part])
        return _parse_json_response(response.text)
    except Exception as e:
        print(f"[extractor] extract_resume_data failed: {e}")
        return {
            "name": "",
            "skills": [],
            "projects": [],
            "experience": [],
            "education": {
                "degree": "", "branch": "", "institution": "", "year": ""
            },
        }


# ── JD extraction (text only — no vision needed) ──────────────────────────────

JD_EXTRACTION_PROMPT = """
You are a job description analyser. Extract structured requirements from the job description below.

Return ONLY a valid JSON object — no explanation, no markdown fences — with this exact schema:

{{
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill3"],
  "required_experience_years": 2,
  "education_requirement": "e.g. Bachelor's in Computer Science or related field",
  "role_summary": "One sentence describing the role"
}}

Rules:
- required_skills: skills explicitly required or listed under requirements/qualifications
- preferred_skills: skills listed as "good to have", "bonus", "preferred", or "familiar with"
- required_experience_years: the minimum years explicitly stated; use 0 if not mentioned
- Return nothing except the JSON object

Job Description:
{jd_text}
"""


def extract_jd_data(jd_text: str, model) -> dict:
    """
    Parses the job description and returns structured requirements.
    """
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text)
    try:
        response = model.generate_content(prompt)
        return _parse_json_response(response.text)
    except Exception as e:
        print(f"[extractor] extract_jd_data failed: {e}")
        return {
            "required_skills": [],
            "preferred_skills": [],
            "required_experience_years": 0,
            "education_requirement": "",
            "role_summary": "",
        }


# ── Gap analysis ──────────────────────────────────────────────────────────────

GAP_ANALYSIS_PROMPT = """
You are an expert career coach and ATS specialist.
Compare the candidate's structured resume data against the job description
and produce a concise, actionable gap analysis.

Return ONLY a valid JSON object — no explanation, no markdown fences — with this exact schema:

{{
  "overall_verdict": "One sentence: Strong match / Moderate match / Weak match and why",
  "missing_skills": ["skill1", "skill2"],
  "weak_sections": [
    {{
      "section": "Skills / Experience / Projects / Education",
      "issue": "What is missing or weak",
      "suggestion": "Specific, actionable thing the candidate should do"
    }}
  ],
  "strengths": ["strength1", "strength2"],
  "quick_wins": ["Reword X bullet to mention Y", "Add a project demonstrating Z"]
}}

Rules:
- missing_skills: JD required skills that are absent or not demonstrated anywhere in the resume
- quick_wins: 2–4 concrete edits the candidate could make today — not generic advice
- Be honest and specific
- Return nothing except the JSON object

Job Description:
{jd_text}

Candidate Resume (structured):
{resume_json}
"""


def generate_gap_analysis(resume_data: dict, jd_text: str, model) -> dict:
    """
    Produces structured gap analysis comparing the resume against the JD.
    """
    prompt = GAP_ANALYSIS_PROMPT.format(
        jd_text=jd_text,
        resume_json=json.dumps(resume_data, indent=2),
    )
    try:
        response = model.generate_content(prompt)
        return _parse_json_response(response.text)
    except Exception as e:
        print(f"[extractor] generate_gap_analysis failed: {e}")
        return {
            "overall_verdict": "Analysis unavailable",
            "missing_skills": [],
            "weak_sections": [],
            "strengths": [],
            "quick_wins": [],
        }