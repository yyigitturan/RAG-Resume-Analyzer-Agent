import os
from docling.document_converter import DocumentConverter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import json
import pandas as pd
import csv
import streamlit as st
from langchain_ollama import ChatOllama

def load_llm(id_model: str, temperature: float): 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    return llm 

def load_local_llm(model, temperature): 
    llm = ChatOllama( 
        model=model, 
        temperature=temperature
    ) 

    return llm 


def format_res(res, return_thinking=False):
  res = res.strip()

  if return_thinking:
    res = res.replace("<think>", "[thinking...] ")
    res = res.replace("</think>", "\n---\n")

  else:
    if "</think>" in res:
      res = res.split("</think>")[-1].strip()

  return res


def parse_doc(file_path):
  converter = DocumentConverter()
  result = converter.convert(file_path)
  content = result.document.export_to_markdown()
  return content


def parse_res_llm(response_text: str, required_fields: list) -> dict:
    try:
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

    except json.JSONDecodeError:
        #Error interpreting model response
        return


def save_json_cv(new_data, path_json, key_name="name"):
    # Load existing JSON, if any
    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    if isinstance(data, dict):
        data = [data]

    # Check if there is already a resume for this person
    candidates = [entry.get(key_name) for entry in data]
    if new_data.get(key_name) in candidates:
        st.warning(f"Resume '{new_data.get(key_name)}' already registered. Ignoring it.")
        return

    # Add and Save
    data.append(new_data)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_cv(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def show_cv_result(result: dict):
    md = f"### 📄 Curriculum Analysis and Summary\n"
    if "name" in result:
        md += f"- **Name:** {result['name']}\n"
    if "area" in result:
        md += f"- **Area:** {result['area']}\n"
    if "skills" in result:
        md += f"- **Skills:** {', '.join(result['skills'])}\n"
    if "summary" in result:
        md += f"- **Summary:** {result['summary']}\n"
    if "interview_questions" in result:
        md += f"- **Interview questions:**\n"
        md += "\n".join([f"  - {q}" for q in result["interview_questions"]]) + "\n"
    if "strengths" in result:
        md += f"- **Strengths:**\n"
        md += "\n".join([f"  - {s}" for s in result["strengths"]]) + "\n"
    if "areas_for_development" in result:
        md += f"- **Areas for development:**\n"
        md += "\n".join([f"  - {a}" for a in result["areas_for_development"]]) + "\n"
    if "important_considerations" in result:
        md += f"- **Important considerations:**\n"
        md += "\n".join([f"  - {i}" for i in result["important_considerations"]]) + "\n"
    if "final_recommendations" in result:
        md += f"- **Final recommendations:** {result['final_recommendations']}\n"
    return md


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

def process_cv(schema, job_details, prompt_template, prompt_score, llm, file_path):

  if file_path:
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"File not found: {file_path}")

  content = parse_doc(file_path)

  chain = prompt_template | llm
  output = chain.invoke({"schema": schema, "cv": content, "job": job_details, "prompt_score": prompt_score})

  res = format_res(output.content)

  return output, res


def display_json_table(path_json):
  with open(path_json, "r", encoding="utf-8") as f:
    data = json.load(f)

  df = pd.DataFrame(data)
  return df