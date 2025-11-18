import os
import json
import re
import csv
import pandas as pd
from langchain_ollama import ChatOllama
from docling.document_converter import DocumentConverter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import display, Markdown



def load_local_llm(model, temperature): 
    llm = ChatOllama( 
        model=model, 
        temperature=temperature
    ) 
    return llm

def load_llm(id_model: str, temperature: float): 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm 

def clean_llm_response(response, return_thinking=False):
  response = response.strip()

  if return_thinking:
    response = response.replace("<think>", "[thinking...] ")
    response = response.replace("</think>", "\n---\n")

  else:
    if "</think>" in response:
      response = response.split("</think>")[-1].strip()

  return response 

def parse_document_to_markdown(file_path):
  converter = DocumentConverter()
  result = converter.convert(file_path)
  content = result.document.export_to_markdown()
  return content

def extract_structured_json(response_text: str, required_fields: list) -> dict:
        #  Remove the reasoning part (<think>...</think>)
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()

        # Locates JSON and parses it
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise json.JSONDecodeError("No JSON found in response", response_text, 0)

        json_str = response_text[start_idx:end_idx]
        info_cv = json.loads(json_str)

        for field in required_fields:
            if field not in info_cv:
                info_cv[field] = []

        return info_cv 

def save_json_cv(new_data, path_json, key_name="name"):
    """
    Appends new resume data (new_data) to an existing JSON file, 
    preventing duplicate entries based on the key_name (default is "name").
    """
    
    if os.path.exists(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If the file is empty or corrupted, start with an empty list.
            data = []
        
    else:
        # If the file does not exist, start with an empty list.
        data = []

    # Ensure data is a list, even if the file only contained a single dictionary.
    if isinstance(data, dict):
        data = [data]
    
    # Check for duplicates
    # Filter out non-dictionary entries just in case of file corruption.
    candidates = [entry.get(key_name) for entry in data if isinstance(entry, dict)]
    
    candidate_id = new_data.get(key_name)
    
    if candidate_id in candidates:
        print(f"WARNING: Resume '{candidate_id}' already registered. Ignoring addition.")
        return

    # Add new data and save
    data.append(new_data)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)    

def load_json_cv(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)        
    
def show_cv_result(result: dict): 
    """
    Displays the resume analysis results (dictionary) in a professional, 
    structured Markdown report format using all specified fields, 
    including new analysis fields (Strengths, Gaps, Recommendations).
    """
    
    # --- Report Header ---
    candidate_name = result.get('name', 'Candidate Name Not Found')
    md = f"## 📄 Candidate Resume Analysis Report: {candidate_name}\n"
    md += f"--- \n"
    
    # --- 1. Core Data ---
    md += f"### 1. Core Profile and Background\n"
    md += f"| Criterion | Value |\n"
    md += f"| :--- | :--- |\n"
    md += f"| **Candidate Name** | **{candidate_name}** |\n"
    md += f"| **Primary Area** | {result.get('area', 'N/A')} |\n"
    md += f"| **Education Summary** | {result.get('education', 'Not specified')} |\n"
    md += "\n"
    
    # --- 2. Executive Summary ---
    md += f"### 2. Executive Summary\n"
    md += f"> {result.get('summary', 'Summary not extracted.')}\n\n"
    
    # --- 3. Technical Skills ---
    md += f"### 3. Technical and Core Skills\n"
    skills = result.get('skills', [])
    if isinstance(skills, list) and skills:
        md += f"**Key Skills:** {', '.join([f'**{s}**' for s in skills])}\n\n"
    else:
        md += "* Skills section was not clearly structured or is empty.\n\n"


    # --- 4. Project Achievements ---
    md += f"### 4. Project Achievements and Impact\n"
    projects = result.get('projects', [])
    
    if projects and isinstance(projects, list):
        for idx, project in enumerate(projects):
            title = project.get('project_title', f"Project {idx+1}")
            summary = project.get('achievements_summary', 'No achievement summary provided.')
            
            md += f"#### 4.{idx+1}. {title}\n"
            md += f"* **Achievement Summary:** {summary}\n\n"
    else:
        md += "* Detailed project achievements or specific metrics were not found.\n\n"

    
    # ---  Alignment and Development Analysis  ---
    md += f"### 5. Fit & Development Analysis\n"
    
    # Strengths
    md += f"#### 5.1. Core Strengths\n"
    strengths = result.get('strengths', [])
    if isinstance(strengths, list) and strengths:
        for item in strengths:
            md += f"* **✅ {item}**\n"
    else:
        md += "* No specific strengths listed.\n"

    # Areas for Development (Gaps)
    md += f"#### 5.2. Areas for Development (Gaps/Risks)\n"
    areas_for_development = result.get('areas_for_development', [])
    if isinstance(areas_for_development, list) and areas_for_development:
        for item in areas_for_development:
            md += f"* **❌ {item}**\n"
    else:
        md += "* No immediate development areas identified.\n"
        
    # Important Considerations
    md += f"#### 5.3. Important Considerations\n"
    important_considerations = result.get('important_considerations', [])
    if isinstance(important_considerations, list) and important_considerations:
        for item in important_considerations:
            md += f"* ⚠️ {item}\n"
    else:
        md += "* No specific notes or verification required.\n"
    md += "\n"


    # --- Interview Focus and Final Recommendation ---
    
    # Interview Question
    md += f"### 6. Next Steps\n"
    md += f"#### Suggested Interview Question\n"
    md += f"💡 **Question:** {result.get('interview_questions', 'No specific question suggested.')}\n\n"
    
    # Final Recommendation
    md += f"#### Final Hiring Recommendation\n"
    md += f"**Overall Fit:** **{result.get('final_recommendation', 'Assessment Missing').upper()}**\n"
    md += f"*(Recommendation is based on fit against the provided Job Description.)*\n"

    
    # --- Display the Markdown Report ---
    display(Markdown(md))

def save_job_to_csv(data, filename):
    headers = ['title', 'description', 'details']
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)                

def load_job(csv_path):
  try:
    
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    
    if 'title' not in df.columns:
        raise ValueError(f"Title not found.")
        
    job = df.iloc[-1]
    
    prompt_text = f"""
    **Job for {job['title']}**

    **Description:**
    {job['description']}

    **Full Details:**
    {job['details']}
    """

    return prompt_text.strip()

  except FileNotFoundError:
    return "Error: Job file not found"
  except (ValueError, IndexError, KeyError) as e:
    return f"Error: CSV data could not be parsed. Please delete the file and try again. Detail: {e}"    

def format_result(result):
    cleaned_string = result.strip()
    
    if cleaned_string.startswith("'") and cleaned_string.endswith("'"):
        cleaned_string = cleaned_string[1:-1].strip()
    
    match = re.search(r'\{.*\}', cleaned_string, re.DOTALL)
    
    if match:
        final_json_string = match.group(0)
    else:
        final_json_string = cleaned_string

    return json.loads(final_json_string)      

def display_json_table(path_json):
  with open(path_json, "r", encoding="utf-8") as f:
    data = json.load(f)

  df = pd.DataFrame(data)
  return df

def json_repair(raw_text):
    text = raw_text.strip()

    if text.startswith("{") and not text.endswith("}"):
        text += "}"

    return text.strip()

def process_cv_raw(schema, job_details, prompt_template, prompt_score, llm, file_path):

    content = parse_document_to_markdown(file_path) 
    chain = prompt_template | llm
    output = chain.invoke({"schema": schema, "resume": content, "job": job_details, "prompt_score": prompt_score})
    response = output.content
    return output, response 


