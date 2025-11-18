from langchain_core.prompts import ChatPromptTemplate


schema = """
{
  "name": "Full name of the candidate",
  "area": "Primary area or field. Classify in only one of these: IT, Marketing, Sales, Financial, Administrative, Others. ",
  "summary": "Concise summary of the candidate",
  "skills": ["skill 1", "skill 2", "..."],
  "education": "Summary of the most relevant academic background",
  "interview_questions": ["At least 3 useful interview question based on the resume, to clarify or explore a specific point"],
  "strengths": ["Key strengths that suggest alignment with the desired role, such as relevant experience or valuable skills"],
  "areas_for_development": ["Points that indicate potential gaps or risks, like missing experience or unlisted technologies"],
  "important_considerations": ["Specific notes that require extra attention or verification, such as missing key information"],
  "final_recommendations": "Final summary with suggestion of next steps (e.g., proceed to interview, request clarification, or consider for a different role)"
  "score": 0.0
}
"""

fields = [
    "name",
    "area",
    "summary",
    "skills",
    "education",
    "interview_questions",
    "strengths",
    "areas_for_development",
    "important_considerations",
    "final_recommendations",
    "score"
]

prompt_score = """
Based on the specific job opening, calculate a final score from 0.0 to 10.0.
The output for this field must include only the final score (x.x), with no additional text or comments.
Be fair and rigorous in your evaluation. A perfect score of 10.0 should only be assigned to candidates who clearly exceed all expectations for the role.

Evaluation criteria:
1. Experience (Weight: 35%) – Analyze previous roles, total time of experience, and how closely they match the job responsibilities.
2. Technical Skills (Weight: 25%) – Check how well the candidate's technical skills align with the job requirements.
3. Education (Weight: 15%) – Assess the relevance of degrees or certifications for the position, including institutions and years of study.
4. Strengths (Weight: 15%) – Evaluate how well the candidate's strengths support the role.
5. Weaknesses (Penalty up to 10%) – Consider the severity of any weaknesses or misalignments with the job requirements.
"""

prompt_template = ChatPromptTemplate.from_template("""
You are a Human Resources specialist with extensive experience in resume analysis.
Your task is to review the content below and extract the information using the exact JSON structure provided.
Respond with only the structured JSON output and use only the specified keys.
Make sure the key names match the schema exactly.
Do not include explanations or comments outside the JSON.

Schema:
{schema}

---

Rules for calculate the score:
{prompt_score}

---

Resume to be analyzed:
'{resume}'

---

Job description:
'{job}'

""")
