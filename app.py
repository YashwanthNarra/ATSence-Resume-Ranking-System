"""
app.py
Streamlit UI — Resume Intelligence System

Flow:
  1. User enters Gemini API key (sidebar)
  2. User uploads one or more PDF resumes + pastes JD
  3. On "Analyse" → runs full pipeline for each resume
  4. Shows ranked table (rank, name, ATS score)
  5. User clicks a resume row → detail panel opens below
     showing score breakdown + matched skills + projects
     + experience + education + gap analysis
"""

import streamlit as st
from extractor import (
    get_gemini_model,
    extract_resume_data,
    extract_jd_data,
    generate_gap_analysis,
)
from scorer import analyse_resume


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume Intelligence System",
    page_icon="📄",
    layout="wide",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.resume-row {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 20px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 8px;
    background: rgba(255,255,255,0.03);
}
.rank-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px; font-weight: 700;
    color: #6366f1; min-width: 32px; text-align: center;
}
.rank-badge.gold   { color: #f59e0b; }
.rank-badge.silver { color: #94a3b8; }
.rank-badge.bronze { color: #b45309; }
.resume-name { flex: 1; font-weight: 500; font-size: 15px; }
.ats-pill {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 15px;
    padding: 4px 14px; border-radius: 99px;
}
.ats-high   { background: rgba(16,185,129,0.12); color: #10b981; }
.ats-mid    { background: rgba(245,158,11,0.12);  color: #f59e0b; }
.ats-low    { background: rgba(239,68,68,0.12);   color: #ef4444; }

.score-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.score-card .label {
    font-size: 12px; color: rgba(148,163,184,0.8);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}
.score-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700;
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6366f1;
    margin: 24px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

.chip-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip {
    font-size: 12px; padding: 3px 10px; border-radius: 99px;
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc; background: rgba(99,102,241,0.08);
}
.chip.missing { border-color: rgba(239,68,68,0.3); color: #fca5a5; background: rgba(239,68,68,0.08); }
.chip.strength { border-color: rgba(16,185,129,0.3); color: #6ee7b7; background: rgba(16,185,129,0.08); }

.gap-card {
    border-left: 3px solid #6366f1; padding: 10px 14px;
    margin-bottom: 10px; border-radius: 0 8px 8px 0;
    background: rgba(99,102,241,0.06);
}
.gap-card .g-section { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.gap-card .g-issue   { font-size: 13px; color: #94a3b8; margin-bottom: 4px; }
.gap-card .g-suggest { font-size: 13px; color: #c7d2fe; }

.quick-win {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.1);
    font-size: 13px;
}
.quick-win:last-child { border-bottom: none; }
.qw-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #6366f1; flex-shrink: 0; margin-top: 5px;
}

.exp-item, .proj-item { padding: 12px 0; border-bottom: 1px solid rgba(128,128,128,0.1); }
.exp-item:last-child, .proj-item:last-child { border-bottom: none; }
.exp-role  { font-weight: 600; font-size: 14px; }
.exp-meta  { font-size: 12px; color: #94a3b8; margin: 2px 0 6px; }
.exp-desc  { font-size: 13px; line-height: 1.6; }
.proj-title { font-weight: 600; font-size: 14px; }
.proj-rel  {
    font-size: 11px; color: #a5b4fc;
    background: rgba(99,102,241,0.1);
    padding: 2px 8px; border-radius: 99px;
    display: inline-block; margin: 3px 0 6px;
}
.proj-tech { font-size: 11px; color: #94a3b8; margin-top: 4px; }

.verdict {
    padding: 10px 16px; border-radius: 8px;
    font-weight: 500; font-size: 14px; margin-bottom: 16px;
}
.verdict.strong   { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.verdict.moderate { background: rgba(245,158,11,0.1);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }
.verdict.weak     { background: rgba(239,68,68,0.1);   color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }

.hdivider { border: none; border-top: 1px solid rgba(128,128,128,0.12); margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ats_pill_cls(score):
    if score >= 70: return "ats-high"
    if score >= 45: return "ats-mid"
    return "ats-low"

def score_color(score):
    if score >= 70: return "#10b981"
    if score >= 45: return "#f59e0b"
    return "#ef4444"

def rank_badge_cls(rank):
    return {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "")

def verdict_cls(text):
    t = text.lower()
    if "strong" in t: return "strong"
    if "weak"   in t: return "weak"
    return "moderate"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    _secret_key = st.secrets.get("GEMINI_API_KEY", "")
    if _secret_key:
        api_key = _secret_key
        st.success("API key loaded from secrets ✓", icon="🔑")
    else:
        api_key = st.text_input(
            "Gemini API key", type="password", placeholder="AIza…",
            help="Or add GEMINI_API_KEY to .streamlit/secrets.toml",
        )
    st.markdown("---")
    st.markdown(
        "**Scoring weights**\n"
        "- Skills — 50 %\n"
        "- Projects — 25 %\n"
        "- Experience — 15 %\n"
        "- Education — 10 %"
    )
    st.markdown("---")
    st.caption("LLM: gemini-2.0-flash · Embeddings: intfloat/e5-small-v2")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='font-family:Space Grotesk,sans-serif;font-size:28px;"
    "font-weight:700;margin-bottom:4px'>📄 Resume Intelligence System</h1>"
    "<p style='color:#94a3b8;margin-top:0;margin-bottom:24px;font-size:14px'>"
    "Upload resumes · paste a job description · get ranked scores and gap analysis.</p>",
    unsafe_allow_html=True,
)


# ── Inputs ────────────────────────────────────────────────────────────────────

col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF)", type=["pdf"],
        accept_multiple_files=True,
    )
with col_r:
    jd_text = st.text_area("Job description", height=220,
                            placeholder="Paste the full job description here…")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
run_btn = st.button("Analyse Resumes", type="primary")


# ── Session state ─────────────────────────────────────────────────────────────

if "results"      not in st.session_state: st.session_state.results      = []
if "selected_idx" not in st.session_state: st.session_state.selected_idx = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

if run_btn:
    if not api_key:
        st.error("Enter your Gemini API key in the sidebar."); st.stop()
    if not uploaded_files:
        st.warning("Upload at least one resume PDF."); st.stop()
    if not jd_text.strip():
        st.warning("Paste a job description."); st.stop()

    gemini = get_gemini_model(api_key)

    with st.spinner("Parsing job description…"):
        jd_data = extract_jd_data(jd_text, gemini)
    if jd_data.get("role_summary"):
        st.caption(f"Role detected: {jd_data['role_summary']}")

    results   = []
    progress  = st.progress(0, text="Starting…")

    for i, f in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files),
                          text=f"Analysing {f.name} ({i+1}/{len(uploaded_files)})…")

        # Pass PDF directly to Gemini vision - handles any layout/column format
        resume_data = extract_resume_data(f, gemini)
        if not resume_data.get("name") and not resume_data.get("skills"):
            st.warning(f"Could not parse {f.name} — skipped."); continue

        gap    = generate_gap_analysis(resume_data, jd_text, gemini)
        result = analyse_resume(resume_data, jd_data, jd_text, gap)
        result["filename"] = f.name
        results.append(result)

    progress.empty()

    results.sort(key=lambda r: r["ats_score"], reverse=True)
    st.session_state.results      = results
    st.session_state.selected_idx = None
    st.success(f"Done — {len(results)} resume(s) ranked.")


# ── Ranking table ─────────────────────────────────────────────────────────────

results = st.session_state.results

if results:
    st.markdown("<div class='section-label'>Ranking</div>", unsafe_allow_html=True)

    for i, r in enumerate(results):
        display_name = r["name"] if r["name"] else r["filename"]
        is_sel = st.session_state.selected_idx == i

        row_col, btn_col = st.columns([11, 1])
        with row_col:
            pill_cls  = ats_pill_cls(r["ats_score"])
            ats_score = r["ats_score"]
            filename  = r["filename"]
            st.markdown(
                f"<div class='resume-row'>"
                f"  <span class='rank-badge {rank_badge_cls(i+1)}'>#{i+1}</span>"
                f"  <span class='resume-name'>{display_name}"
                f"    <span style='font-size:11px;color:#64748b;margin-left:8px'>{filename}</span>"
                f"  </span>"
                f"  <span class='ats-pill {pill_cls}'>{ats_score}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with btn_col:
            label = "▲ Hide" if is_sel else "▼ View"
            if st.button(label, key=f"row_{i}"):
                st.session_state.selected_idx = None if is_sel else i
                st.rerun()

    # ── Detail panel ──────────────────────────────────────────────────────────

    sel = st.session_state.selected_idx
    if sel is not None and sel < len(results):
        r   = results[sel]
        gap = r.get("gap_analysis", {})

        st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)

        # Candidate header
        st.markdown(
            f"<h2 style='font-family:Space Grotesk,sans-serif;font-size:20px;"
            f"font-weight:700;margin-bottom:2px'>{r['name'] or r['filename']}</h2>"
            f"<p style='color:#64748b;font-size:13px;margin-top:0'>{r['filename']}</p>",
            unsafe_allow_html=True,
        )

        # ── Score breakdown ────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Score breakdown</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in zip(
            [c1, c2, c3, c4, c5],
            ["ATS Score", "Skills", "Projects", "Experience", "Education"],
            [r["ats_score"], r["skill_score"], r["project_score"],
             r["experience_score"], r["education_score"]],
        ):
            col.markdown(
                f"<div class='score-card'>"
                f"  <div class='label'>{label}</div>"
                f"  <div class='value' style='color:{score_color(val)}'>{val}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Gap analysis ───────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Gap analysis</div>", unsafe_allow_html=True)

        verdict = gap.get("overall_verdict", "")
        if verdict:
            st.markdown(
                f"<div class='verdict {verdict_cls(verdict)}'>{verdict}</div>",
                unsafe_allow_html=True,
            )

        gl, gr = st.columns(2, gap="large")

        with gl:
            strengths = gap.get("strengths", [])
            if strengths:
                st.markdown("**Strengths**")
                st.markdown(
                    "<div class='chip-wrap'>"
                    + "".join(f"<span class='chip strength'>{s}</span>" for s in strengths)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            missing = gap.get("missing_skills", [])
            if missing:
                st.markdown("**Missing skills**")
                st.markdown(
                    "<div class='chip-wrap'>"
                    + "".join(f"<span class='chip missing'>{s}</span>" for s in missing)
                    + "</div>",
                    unsafe_allow_html=True,
                )

        with gr:
            weak = gap.get("weak_sections", [])
            if weak:
                st.markdown("**Weak sections**")
                html = ""
                for ws in weak:
                    html += (
                        f"<div class='gap-card'>"
                        f"  <div class='g-section'>{ws.get('section','')}</div>"
                        f"  <div class='g-issue'>{ws.get('issue','')}</div>"
                        f"  <div class='g-suggest'>→ {ws.get('suggestion','')}</div>"
                        f"</div>"
                    )
                st.markdown(html, unsafe_allow_html=True)

        qw = gap.get("quick_wins", [])
        if qw:
            st.markdown("**Quick wins — things to fix today**")
            items = "".join(
                f"<div class='quick-win'><div class='qw-dot'></div>{w}</div>"
                for w in qw
            )
            st.markdown(
                f"<div style='border:1px solid rgba(99,102,241,0.15);"
                f"border-radius:8px;padding:12px 16px'>{items}</div>",
                unsafe_allow_html=True,
            )

        # ── Matched skills ─────────────────────────────────────────────────
        matched = r.get("matched_skills", [])
        if matched:
            st.markdown("<div class='section-label'>Matched skills</div>", unsafe_allow_html=True)
            def _skill_chip(m):
                return (
                    "<span class='chip' title='Resume: {rs} · sim {sim}'>{jd}</span>".format(
                        rs=m["resume_skill"], sim=m["similarity"], jd=m["jd_skill"]
                    )
                )
            st.markdown(
                "<div class='chip-wrap'>" + "".join(_skill_chip(m) for m in matched) + "</div>",
                unsafe_allow_html=True,
            )

        # ── Projects ───────────────────────────────────────────────────────
        projs = r.get("ranked_projects", [])
        if projs:
            st.markdown("<div class='section-label'>Projects — ranked by relevance</div>",
                        unsafe_allow_html=True)
            html = ""
            for p in projs:
                tech = ", ".join(p.get("technologies", []))
                html += (
                    f"<div class='proj-item'>"
                    f"  <div class='proj-title'>#{p.get('rank','')} {p.get('title','Untitled')}</div>"
                    f"  <span class='proj-rel'>Relevance {round(p['similarity']*100,1)}%</span>"
                    f"  <div class='exp-desc'>{p.get('description','')}</div>"
                    + (f"  <div class='proj-tech'>Tech: {tech}</div>" if tech else "")
                    + "</div>"
                )
            st.markdown(html, unsafe_allow_html=True)

        # ── Experience ─────────────────────────────────────────────────────
        exp_list = r.get("experience", [])
        if exp_list:
            st.markdown(
                f"<div class='section-label'>Experience "
                f"<span style='font-size:11px;font-weight:400;text-transform:none;"
                f"color:#64748b'>{r.get('total_experience_years', 0)} yrs total</span></div>",
                unsafe_allow_html=True,
            )
            html = ""
            for e in exp_list:
                html += (
                    f"<div class='exp-item'>"
                    f"  <div class='exp-role'>{e.get('role','')}</div>"
                    f"  <div class='exp-meta'>{e.get('company','')} · "
                    f"    {e.get('duration_text','')} ({e.get('duration_years',0)} yrs)</div>"
                    f"  <div class='exp-desc'>{e.get('description','')}</div>"
                    f"</div>"
                )
            st.markdown(html, unsafe_allow_html=True)

        # ── Education ──────────────────────────────────────────────────────
        edu = r.get("education", {})
        if any(edu.values()):
            st.markdown("<div class='section-label'>Education</div>", unsafe_allow_html=True)
            year_str = f" · {edu['year']}" if edu.get("year") else ""
            st.markdown(
                f"<div style='font-size:14px'>"
                f"  <strong>{edu.get('degree','')} in {edu.get('branch','')}</strong><br>"
                f"  <span style='color:#94a3b8'>{edu.get('institution','')}{year_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)