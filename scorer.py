"""
scorer.py
All scoring logic:
  - semantic_skill_match()     → match resume skills vs JD skills via embeddings
  - score_projects()           → rank projects by cosine similarity to JD
  - calculate_experience_score()
  - calculate_education_score()
  - calculate_ats_score()      → final weighted ATS score
  - analyse_resume()           → full pipeline for one resume, returns complete result dict
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Embedding model (loaded once at import time) ─────────────────────────────

_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("intfloat/e5-small-v2")
    return _embed_model


# ── Semantic skill matching ───────────────────────────────────────────────────

def semantic_skill_match(
    resume_skills: list[str],
    jd_skills: list[str],
    threshold: float = 0.82,
) -> list[dict]:
    """
    For every JD skill, finds the best-matching resume skill via cosine similarity.
    Returns only matches above `threshold`.

    Returns:
        [{"jd_skill": str, "resume_skill": str, "similarity": float}, ...]
    """
    if not resume_skills or not jd_skills:
        return []

    model = get_embed_model()
    resume_emb = model.encode(resume_skills, normalize_embeddings=True)
    jd_emb = model.encode(jd_skills, normalize_embeddings=True)

    # similarity matrix: shape (len(resume_skills), len(jd_skills))
    sim_matrix = cosine_similarity(resume_emb, jd_emb)

    matched = []
    for j, jd_skill in enumerate(jd_skills):
        best_i = int(np.argmax(sim_matrix[:, j]))
        best_score = float(sim_matrix[best_i, j])
        if best_score >= threshold:
            matched.append({
                "jd_skill": jd_skill,
                "resume_skill": resume_skills[best_i],
                "similarity": round(best_score, 3),
            })

    return matched


# ── Project scoring ───────────────────────────────────────────────────────────

def score_projects(projects: list[dict], jd_text: str) -> list[dict]:
    """
    Ranks projects by semantic relevance to the full JD text.
    Each project dict must have at least a "description" key.
    Adds "similarity" and "rank" to each entry.

    Returns:
        Sorted list of project dicts (highest similarity first).
    """
    if not projects or not jd_text.strip():
        return []

    model = get_embed_model()

    # Build one string per project: title + description + technologies
    def project_text(p: dict) -> str:
        parts = [p.get("title", ""), p.get("description", "")]
        techs = p.get("technologies", [])
        if techs:
            parts.append(" ".join(techs))
        return " ".join(filter(None, parts))

    project_strings = [project_text(p) for p in projects]

    project_emb = model.encode(project_strings, normalize_embeddings=True)
    jd_emb = model.encode([jd_text], normalize_embeddings=True)

    scores = cosine_similarity(project_emb, jd_emb).flatten()

    ranked = []
    for i, p in enumerate(projects):
        ranked.append({**p, "similarity": round(float(scores[i]), 3)})

    ranked.sort(key=lambda x: x["similarity"], reverse=True)
    for i, p in enumerate(ranked):
        p["rank"] = i + 1

    return ranked


# ── Component score helpers ───────────────────────────────────────────────────

def _skill_score(matched: list[dict], total_jd_skills: int) -> float:
    if total_jd_skills == 0:
        return 0.0
    return round(len(matched) / total_jd_skills * 100, 2)


def _project_score(ranked_projects: list[dict]) -> float:
    if not ranked_projects:
        return 0.0
    top = ranked_projects[:3]
    avg = sum(p["similarity"] for p in top) / len(top)
    return round(avg * 100, 2)


def _experience_score(total_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 100.0
    return round(min(total_years / required_years, 1.0) * 100, 2)


def _education_score(resume_edu: dict, jd_edu_requirement: str, model) -> float:
    """
    Uses embedding similarity between the candidate's degree+branch
    and the JD education requirement string.
    Falls back to 50.0 if either side is empty.
    """
    candidate_edu = " ".join(filter(None, [
        resume_edu.get("degree", ""),
        resume_edu.get("branch", ""),
    ]))
    if not candidate_edu or not jd_edu_requirement:
        return 50.0

    embed = get_embed_model()
    embs = embed.encode([candidate_edu, jd_edu_requirement], normalize_embeddings=True)
    score = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
    return round(score * 100, 2)


# ── Weighted ATS score ────────────────────────────────────────────────────────

WEIGHTS = {
    "skills":     0.50,
    "projects":   0.25,
    "experience": 0.15,
    "education":  0.10,
}


def calculate_ats_score(
    skill_score: float,
    project_score: float,
    experience_score: float,
    education_score: float,
) -> float:
    return round(
        skill_score     * WEIGHTS["skills"]     +
        project_score   * WEIGHTS["projects"]   +
        experience_score * WEIGHTS["experience"] +
        education_score * WEIGHTS["education"],
        2,
    )


# ── Full single-resume pipeline ───────────────────────────────────────────────

def analyse_resume(
    resume_data: dict,
    jd_data: dict,
    jd_text: str,
    gap_analysis: dict,
) -> dict:
    """
    Runs all scoring steps and returns a complete result dict for one resume.

    Args:
        resume_data:   Output of extractor.extract_resume_data()
        jd_data:       Output of extractor.extract_jd_data()
        jd_text:       Raw job description string
        gap_analysis:  Output of extractor.generate_gap_analysis()

    Returns:
        Full result dict ready for the UI.
    """
    # Combine required + preferred JD skills for matching
    all_jd_skills = list(dict.fromkeys(
        jd_data.get("required_skills", []) + jd_data.get("preferred_skills", [])
    ))
    resume_skills = resume_data.get("skills", [])

    matched_skills = semantic_skill_match(resume_skills, all_jd_skills)
    skill_score = _skill_score(matched_skills, len(all_jd_skills))

    ranked_projects = score_projects(resume_data.get("projects", []), jd_text)
    project_score = _project_score(ranked_projects)

    experience_list = resume_data.get("experience", [])
    total_years = sum(e.get("duration_years", 0.0) for e in experience_list)
    required_years = jd_data.get("required_experience_years", 0)
    experience_score = _experience_score(total_years, required_years)

    education_score = _education_score(
        resume_data.get("education", {}),
        jd_data.get("education_requirement", ""),
        get_embed_model(),
    )

    ats_score = calculate_ats_score(
        skill_score, project_score, experience_score, education_score
    )

    return {
        "name":             resume_data.get("name", ""),
        "ats_score":        ats_score,
        "skill_score":      skill_score,
        "project_score":    project_score,
        "experience_score": experience_score,
        "education_score":  education_score,
        "matched_skills":   matched_skills,
        "ranked_projects":  ranked_projects,
        "experience":       experience_list,
        "education":        resume_data.get("education", {}),
        "total_experience_years": round(total_years, 1),
        "gap_analysis":     gap_analysis,
        "jd_role_summary":  jd_data.get("role_summary", ""),
    }