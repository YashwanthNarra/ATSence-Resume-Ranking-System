"""
scorer.py
All scoring logic:
  - match_skills()      → soft semantic similarity per JD skill (no hard threshold)
  - score_projects()    → ranks projects against role summary + required skills only
  - score_experience()  → compares years of experience
  - score_education()   → compares candidate degree vs JD requirement
  - final_ats_score()   → weighted combination of all scores
  - analyse_resume()    → runs everything, returns one big result dict
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embed_model = SentenceTransformer("intfloat/e5-small-v2")

WEIGHTS = {
    "skills":     0.50,
    "projects":   0.25,
    "experience": 0.15,
    "education":  0.10,
}


def match_skills(resume_skills: list, jd_skills: list) -> tuple[list, float]:
    """
    Soft scoring: every JD skill gets a similarity score (0-1) against its
    closest resume skill. Skill score = average across all JD skills.
    No hard threshold — partial matches contribute proportionally.

    Returns:
        matches   : list of dicts showing the best resume match per JD skill
        skill_score: float 0-100
    """
    if not jd_skills:
        return [], 0.0
    if not resume_skills:
        # No resume skills at all — every JD skill scores 0
        return [{"jd_skill": s, "resume_skill": None, "similarity": 0.0} for s in jd_skills], 0.0

    resume_emb = embed_model.encode(resume_skills, normalize_embeddings=True)
    jd_emb     = embed_model.encode(jd_skills,     normalize_embeddings=True)
    sim_matrix = cosine_similarity(resume_emb, jd_emb)  # shape: (n_resume, n_jd)

    matches = []
    total_similarity = 0.0

    for j, jd_skill in enumerate(jd_skills):
        best_i     = int(np.argmax(sim_matrix[:, j]))
        best_score = float(sim_matrix[best_i, j])
        total_similarity += best_score
        matches.append({
            "jd_skill":     jd_skill,
            "resume_skill": resume_skills[best_i],
            "similarity":   round(best_score, 3),
        })

    skill_score = round((total_similarity / len(jd_skills)) * 100, 2)
    return matches, skill_score


def score_projects(projects: list, role_summary: str, required_skills: list) -> list:
    """
    Score each project against role summary + required skills only.
    Avoids noise from JD boilerplate (benefits, location, company description).

    Blended signal per project:
      70% skill overlap  (project technologies vs required_skills)
      30% semantic similarity against role summary + skills string
    """
    if not projects:
        return []

    # Build the focused JD signal: role summary + required skills
    skills_str  = " ".join(required_skills) if required_skills else ""
    focused_jd  = " ".join(filter(None, [role_summary, skills_str])).strip()

    if not focused_jd:
        # Nothing useful to score against — rank by order, score 0
        return [{**p, "similarity": 0.0, "rank": i + 1} for i, p in enumerate(projects)]

    def project_to_text(p):
        parts = [p.get("title", ""), p.get("description", "")]
        techs = p.get("technologies", [])
        if techs:
            parts.append(" ".join(techs))
        return " ".join(x for x in parts if x)

    project_texts = [project_to_text(p) for p in projects]
    project_emb   = embed_model.encode(project_texts, normalize_embeddings=True)
    jd_emb        = embed_model.encode([focused_jd],  normalize_embeddings=True)
    semantic_scores = cosine_similarity(project_emb, jd_emb).flatten()

    # Skill overlap: fraction of required_skills found in project technologies
    required_lower = [s.lower() for s in required_skills]

    def skill_overlap(p):
        if not required_lower:
            return 0.0
        techs = [t.lower() for t in p.get("technologies", [])]
        if not techs:
            return 0.0
        matched = sum(1 for req in required_lower if any(req in t or t in req for t in techs))
        return matched / len(required_lower)

    ranked = []
    for i, p in enumerate(projects):
        overlap   = skill_overlap(p)
        semantic  = float(semantic_scores[i])
        blended   = round(0.70 * overlap + 0.30 * semantic, 3)
        ranked.append({**p, "similarity": blended})

    ranked.sort(key=lambda x: x["similarity"], reverse=True)
    for i, p in enumerate(ranked):
        p["rank"] = i + 1

    return ranked


def score_experience(total_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 100.0
    return round(min(total_years / required_years, 1.0) * 100, 2)


def score_education(edu: dict, jd_edu_requirement: str) -> float:
    candidate_text = " ".join(filter(None, [edu.get("degree", ""), edu.get("branch", "")]))
    if not candidate_text or not jd_edu_requirement:
        return 50.0
    embs  = embed_model.encode([candidate_text, jd_edu_requirement], normalize_embeddings=True)
    score = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
    return round(score * 100, 2)


def final_ats_score(skill_score, project_score, experience_score, education_score) -> float:
    return round(
        skill_score      * WEIGHTS["skills"]     +
        project_score    * WEIGHTS["projects"]   +
        experience_score * WEIGHTS["experience"] +
        education_score  * WEIGHTS["education"],
        2,
    )


def analyse_resume(resume_data: dict, jd_data: dict, jd_text: str, gap: dict) -> dict:
    all_jd_skills  = list(dict.fromkeys(
        jd_data.get("required_skills", []) + jd_data.get("preferred_skills", [])
    ))
    resume_skills          = resume_data.get("skills", [])
    matched_skills, skill_score = match_skills(resume_skills, all_jd_skills)

    ranked_projects = score_projects(
        resume_data.get("projects", []),
        jd_data.get("role_summary", ""),
        jd_data.get("required_skills", []),
    )
    project_score = 0.0
    if ranked_projects:
        top2          = ranked_projects[:2]
        project_score = round(sum(p["similarity"] for p in top2) / len(top2) * 100, 2)

    experience_list  = resume_data.get("experience", [])
    total_years      = sum(e.get("duration_years", 0.0) for e in experience_list)
    required_years   = jd_data.get("required_experience_years", 0)
    experience_score = score_experience(total_years, required_years)

    education_score = score_education(
        resume_data.get("education", {}),
        jd_data.get("education_requirement", ""),
    )

    ats_score = final_ats_score(skill_score, project_score, experience_score, education_score)

    return {
        "name":                   resume_data.get("name", ""),
        "ats_score":              ats_score,
        "skill_score":            skill_score,
        "project_score":          project_score,
        "experience_score":       experience_score,
        "education_score":        education_score,
        "matched_skills":         matched_skills,
        "ranked_projects":        ranked_projects,
        "experience":             experience_list,
        "education":              resume_data.get("education", {}),
        "total_experience_years": round(total_years, 1),
        "gap_analysis":           gap,
        "role_summary":           jd_data.get("role_summary", ""),
    }
