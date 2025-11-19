import streamlit as st
import uuid
from src.helper import *
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Resume Screening and Analysis", page_icon="📄", layout="wide")

id_model = "llama3"
temperature = 0.0
json_file = 'resumes.json'
path_job_csv = "jobs.csv"

llm = load_local_llm(id_model, temperature)

job = {}
job['title'] = "Full Stack Developer"
job['description'] = """
We are looking for a Full Stack Developer to join our tech team and contribute to strategic projects
focused on scalable, data-driven solutions. The professional will be responsible for developing, maintaining,
and evolving robust web applications, while collaborating with cross-functional teams to continuously deliver business value.
"""

job['details'] = """
Responsibilities:
- Develop and maintain modern web applications using both front-end and back-end technologies.
- Collaborate with product, UX, and data teams to understand requirements and propose effective solutions.
- Build APIs, integrations, and interactive dashboards.
- Ensure best practices in version control, testing, and documentation.
- Participate in code reviews, deployments, and ongoing architectural improvements.

Requirements:
- Strong knowledge of Python, JavaScript, and SQL.
- Hands-on experience with frameworks such as React, Node.js, and Django.
- Familiarity with version control using Git.
- Experience with cloud services like AWS or Google Cloud Platform.
- Team player with strong communication skills and a collaborative mindset.

Desirable qualifications:
- Experience with Power BI or other data visualization tools.
- Background in Agile methodologies (Scrum, Kanban).
- Personal projects, open-source contributions, or a technical portfolio.
- Cloud certifications or relevant credentials in software engineering.
"""

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
'{cv}'

---

Job description:
'{job}'

""")

if "uploader_key" not in st.session_state:
  st.session_state.uploader_key = str(uuid.uuid4())

if "selected_cv" not in st.session_state:
  st.session_state.selected_cv = None

save_job_to_csv(job, path_job_csv)
job_details = load_job(path_job_csv)

col1, col2 = st.columns(2)
with col1:
  st.header("📄 Resume Screening and Analysis")
  st.markdown("##### Job position: {}".format(job["title"]))
with col2:
  uploaded_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"], key=st.session_state.uploader_key)

if uploaded_file is not None:
  with st.spinner("Analyzing the resume..."):
    path = uploaded_file.name
    with open(path, "wb") as f:
      f.write(uploaded_file.read())

    output, res = process_cv(schema, job_details, prompt_template, prompt_score, llm, path)
    structured_data = parse_res_llm(res, fields)
    save_json_cv(structured_data, path_json=json_file, key_name="name")

    st.success("Resume successfully analyzed!")
    st.session_state.uploader_key = str(uuid.uuid4())

  st.write(show_cv_result(structured_data))

  with st.expander("View structured data (JSON)"):
    st.json(structured_data)

if os.path.exists(json_file):

    st.subheader("List of resumes analyzed", divider="gray")
    df = display_json_table(json_file)
    # Shows only desired columns
    for i, row in df.iterrows():
        cols = st.columns([1, 3, 1, 5])  # Proportional adjustment between columns

        with cols[0]:
            if st.button("View details", key=f"btn_{i}"):
                st.session_state.selected_cv = row.to_dict()
        with cols[1]:
            st.write(f"**Name:** {row.get('name', '-')}")
        with cols[2]:
            st.write(f"**Score:** {row.get('score', '-')}")
        with cols[3]:
            st.write(f"**Summary:** {row.get('summary', '-')}")

if st.session_state.selected_cv:
  st.markdown("----")
  st.write(show_cv_result(st.session_state.selected_cv))
  with st.expander("View structured data (JSON)"):
    st.json(st.session_state.selected_cv)

if os.path.exists(json_file):
  with open(json_file, "r", encoding="utf-8") as f:
    json_data = f.read()
  st.download_button(
      label="📥 Download .json",
      data=json_data,
      file_name=json_file,
      mime="application/json"
  )

  df = display_json_table(json_file)
  st.dataframe(df)