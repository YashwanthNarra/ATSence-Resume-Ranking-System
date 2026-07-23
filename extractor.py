"""
extractor.py
Handles all Gemini API calls:
  - extract_resume_data()   → reads PDF visually → returns structured dict
  - extract_jd_data()       → pulls required skills/experience from JD text
  - generate_gap_analysis() → compares resume vs JD → returns feedback
"""

import json
import re
import google.generativeai as genai


def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def read_pdf_bytes(pdf_source) -> bytes:
    if hasattr(pdf_source, "read"):
        raw = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        return raw
    with open(pdf_source, "rb") as f:
        return f.read()


def parse_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    return json.loads(cleaned)


RESUME_PROMPT = """
You are a resume parser. The attached file is a resume PDF.
Read it visually including multi-column layouts, styled sections, and skill bars.

Return ONLY a valid JSON object with this exact structure:

{
  "name": "Full name or empty string",
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
      "duration_text": "Jan 2022 - Present",
      "duration_years": 1.5,
      "description": "What they did"
    }
  ],
  "education": {
    "degree": "B.Tech or similar",
    "branch": "Computer Science or similar",
    "institution": "College name",
    "year": "Graduation year or empty string"
  }
}

Rules:
- skills: collect every skill mentioned anywhere on the resume
- duration_years: calculate as a float from duration_text, use 0.0 if unclear
- If any section is missing use empty list or empty string, never skip a key
- Return nothing except the JSON object
"""


def extract_resume_data(pdf_bytes: bytes, model) -> dict:
    pdf_part = {
        "inline_data": {
            "mime_type": "application/pdf",
            "data": pdf_bytes,
        }
    }
    try:
        response = model.generate_content([RESUME_PROMPT, pdf_part])
        return parse_json(response.text)
    except Exception as e:
        print(f"extract_resume_data error: {e}")
        return {
            "name": "",
            "skills": [],
            "projects": [],
            "experience": [],
            "education": {"degree": "", "branch": "", "institution": "", "year": ""},
        }


JD_PROMPT = """
You are a job description analyser.

Return ONLY a valid JSON object:

{{
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill3"],
  "required_experience_years": 2,
  "education_requirement": "Bachelor's in Computer Science or related field",
  "role_summary": "One sentence describing the role"
}}

Rules:
- required_skills: explicitly required skills
- preferred_skills: nice to have or preferred skills
- required_experience_years: minimum years stated, use 0 if not mentioned
- Return nothing except the JSON object

Job Description:
{jd_text}
"""


def extract_jd_data(jd_text: str, model) -> dict:
    prompt = JD_PROMPT.format(jd_text=jd_text)
    try:
        response = model.generate_content(prompt)
        return parse_json(response.text)
    except Exception as e:
        print(f"extract_jd_data error: {e}")
        return {
            "required_skills": [],
            "preferred_skills": [],
            "required_experience_years": 0,
            "education_requirement": "",
            "role_summary": "",
        }


GAP_PROMPT = """
You are a career coach. Compare the resume against the job description.

Return ONLY a valid JSON object:

{{
  "overall_verdict": "Strong match / Moderate match / Weak match and one reason why",
  "missing_skills": ["skill1", "skill2"],
  "weak_sections": [
    {{
      "section": "Skills or Experience or Projects or Education",
      "issue": "What is weak or missing",
      "suggestion": "What the candidate should do"
    }}
  ],
  "strengths": ["strength1", "strength2"],
  "quick_wins": ["Specific thing to fix today"]
}}

Rules:
- missing_skills: JD required skills not found anywhere in the resume
- quick_wins: 2 to 4 concrete edits, not generic advice
- Return nothing except the JSON object

Job Description:
{jd_text}

Resume structured data:
{resume_json}
"""


def generate_gap_analysis(resume_data: dict, jd_text: str, model) -> dict:
    prompt = GAP_PROMPT.format(
        jd_text=jd_text,
        resume_json=json.dumps(resume_data, indent=2),
    )
    try:
        response = model.generate_content(prompt)
        return parse_json(response.text)
    except Exception as e:
        print(f"generate_gap_analysis error: {e}")
        return {
            "overall_verdict": "Analysis unavailable",
            "missing_skills": [],
            "weak_sections": [],
            "strengths": [],
            "quick_wins": [],
        }
