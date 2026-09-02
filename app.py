import streamlit as st
from sentence_transformers import SentenceTransformer, util
import yake

# ============================================
# SETUP (runs once, cached so it doesn't reload every time you interact)
# ============================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()
kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=40, dedupLim=0.9)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_key_terms(text):
    """Pull out likely skill/keyword phrases from text using YAKE."""
    keywords = kw_extractor.extract_keywords(text)
    return {kw.lower().strip() for kw, score in keywords}

def is_real_skill_term(term):
    noise_starters = {"a", "the", "this", "that", "these", "those"}
    generic_words = {"role", "position", "opportunities", "teams", "site"}
    first_word = term.split()[0]
    if first_word in noise_starters:
        return False
    if term in generic_words:
        return False
    return True

def get_missing_skills(resume_text, job_description):
    jd_terms = extract_key_terms(job_description)
    resume_text_lower = resume_text.lower()
    missing = [term for term in jd_terms if term not in resume_text_lower]
    cleaned = sorted([term for term in missing if is_real_skill_term(term)])
    return cleaned

# ============================================
# APP LAYOUT
# ============================================

st.set_page_config(page_title="Resume-JD Matcher", page_icon="🎯")

st.title("🎯 Resume ↔ Job Description Matcher")
st.write(
    "Paste your resume text and a job description below. "
    "This tool uses a pretrained deep learning model (Sentence-BERT) to score how well "
    "they match in **meaning**, not just shared keywords — then highlights skills the "
    "job description mentions that your resume doesn't."
)

col1, col2 = st.columns(2)

with col1:
    resume_text = st.text_area("📄 Your Resume Text", height=250,
                                placeholder="Paste your resume summary, skills, or project description here...")

with col2:
    job_description = st.text_area("💼 Job Description", height=250,
                                    placeholder="Paste the full job description here...")

if st.button("Analyze Match", type="primary"):
    if not resume_text.strip() or not job_description.strip():
        st.warning("Please paste text into both boxes first.")
    else:
        with st.spinner("Analyzing..."):
            embedding_resume = model.encode(resume_text, convert_to_tensor=True)
            embedding_jd = model.encode(job_description, convert_to_tensor=True)
            match_score = util.cos_sim(embedding_resume, embedding_jd).item()

            missing_skills = get_missing_skills(resume_text, job_description)

        st.subheader("Match Score")
        score_percent = match_score * 100
        st.metric(label="Semantic Match", value=f"{score_percent:.1f}%")
        st.progress(max(min(match_score, 1.0), 0.0))

        if score_percent >= 60:
            st.success("Strong semantic match with this role.")
        elif score_percent >= 40:
            st.info("Moderate match — some alignment, some gaps to address.")
        else:
            st.warning("Low semantic match — this role may need different framing or you may want to target other roles.")

        st.subheader("Skills/Terms in the JD Not Found in Your Resume")
        if missing_skills:
            for term in missing_skills:
                st.write(f"- {term}")
        else:
            st.write("No major gaps found — good coverage!")

st.caption("Built with Sentence-BERT (all-MiniLM-L6-v2) for semantic embeddings and YAKE for keyword extraction.")
