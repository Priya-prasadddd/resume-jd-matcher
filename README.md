# 🎯 Resume ↔ Job Description Matcher

A deep learning tool that scores how well a resume matches a job description **by meaning, not just keywords** — and highlights the specific skills/terms the job description mentions that the resume doesn't.

**Live app:** [resume-jd-matcher-iamnsgxsncahazqea8nymd.streamlit.app](https://resume-jd-matcher-iamnsgxsncahazqea8nymd.streamlit.app)

## Why I built this

Most resume-JD matching tools rely on basic keyword overlap — they miss that "led a team" and "management experience" mean the same thing even though they share zero words. I built this to solve my own job search problem: understand *semantic* fit with a role, not just whether the same words appear, and identify concrete gaps to close before applying.

## How it works

1. **Semantic matching** — both the resume and job description are converted into embeddings using a pretrained Sentence-BERT model (`all-MiniLM-L6-v2`). Cosine similarity between the two embeddings gives a match score that reflects *meaning*, not word overlap.
2. **Skill-gap extraction** — the job description is scanned with YAKE (unsupervised keyword extraction) to pull out key phrases, which are then checked against the resume text to surface what's missing.
3. **Interface** — built with Streamlit for a simple two-box input and instant results.

## Tech stack

- Python
- `sentence-transformers` (Sentence-BERT) — semantic embeddings
- YAKE — keyword/skill extraction
- Streamlit — web app + deployment (Streamlit Community Cloud)

## Example results

Tested against two real Data Analyst postings:

| Job Posting | Match Score | Notable gaps flagged |
|---|---|---|
| Bengaluru Data Analyst (dashboards/reporting focus) | 47.1% | reporting, stakeholders, data models, statistical analyses, documentation |
| Data Analyst (business-partnering focus) | 53.2% | create automated reports, data collection processes, provide analytical support, generate actionable insights |

**Insight from testing:** across both postings, the recurring gap was business-facing language (e.g. "actionable insights," "analytical support") rather than technical skills — useful, concrete feedback for improving a resume beyond just "add more keywords."

## Notes / lessons learned

Originally used spaCy for keyword extraction, but its C/Cython dependencies (`thinc`, `blis`) failed to build on Streamlit Community Cloud's servers. Switched to YAKE, a pure-Python keyword extractor, which resolved the deployment issue with no loss of functionality.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
