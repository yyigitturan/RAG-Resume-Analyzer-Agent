# helper.py
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
from typing import Dict, Any, List

# --- LLM YÜKLEME FONKSİYONLARI ---

def load_local_llm(model: str, temperature: float): 
    """Ollama üzerinden yerel LLM'i yükler."""
    llm = ChatOllama( 
        model=model, 
        temperature=temperature
    ) 
    return llm

def load_gemini_llm(id_model: str, temperature: float): 
    """Gemini API üzerinden bulut LLM'i yükler."""
    llm = ChatGoogleGenerativeAI(
        model=id_model, # Parametre doğru kullanılıyor.
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm 

# --- BELGE VE METİN İŞLEME FONKSİYONLARI ---

def parse_document_to_markdown(file_path):
    """PDF gibi bir belgeyi okur ve Markdown içeriğini döndürür."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    content = result.document.export_to_markdown()
    return content

# NOT: clean_llm_response fonksiyonu mantıksal olarak gereksizdir ve kaldırılmıştır.

def json_repair(raw_text: str) -> str:
    """Eksik olabilecek kapanış parantezini (}) ekleyerek ham LLM çıktısını onarır."""
    text = raw_text.strip()
    if text.startswith("{") and not text.endswith("}"):
        text += "}"
    return text.strip()

def format_result(result):
    """Ham LLM çıktısından JSON bloğunu regex ile ayıklar, onarır ve sözlüğe çevirir."""
    cleaned_string = result.strip()
    
    if cleaned_string.startswith("'") and cleaned_string.endswith("'"):
        cleaned_string = cleaned_string[1:-1].strip()
        
    match = re.search(r'\{.*\}', cleaned_string, re.DOTALL)
    
    if match:
        final_json_string = match.group(0)
    else:
        final_json_string = cleaned_string
        
    final_json_string = json_repair(final_json_string) 
    
    # Ek temizlik: Kod bloğu işaretlerini kaldır
    final_json_string = final_json_string.replace('```json', '').replace('```', '').strip()

    return json.loads(final_json_string)

# NOT: extract_structured_json fonksiyonu kaldırılmıştır.


# --- CSV VE JSON VERİ YÖNETİMİ ---

def load_job(csv_path: str) -> str:
    """CSV'deki en son iş ilanını okur ve prompt formatında döndürür."""
    try:
        # Pandas ile okuma ve son satırı alma (Basit ve sağlam yol)
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        
        if 'title' not in df.columns:
            # Düzeltme: Hata mesajı düzeltildi
            raise ValueError("CSV dosyasında 'title' sütunu bulunamadı. Ayırıcı yanlış olabilir.")
            
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
        return "Error: İş İlanı dosyası bulunamadı."
    except Exception as e:
        return f"Error: CSV verisi çözümlenemedi. Detay: {e}" 

def save_job_to_csv(data: Dict[str, str], filename: str):
    """İş tanımı verisini CSV'ye kaydeder."""
    headers = ['title', 'description', 'details']
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data) 


def load_json_cv(path_json: str) -> List[Dict[str, Any]]:
    """JSON dosyasını güvenli bir şekilde yükler."""
    if not os.path.exists(path_json):
        return []
    try:
        with open(path_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_json_cv(new_data: Dict[str, Any], path_json: str, key_name="name"):
    """Yeni özgeçmiş verisini mevcut JSON dosyasına ekler (mükerrer girişi önler)."""
    
    data = load_json_cv(path_json) 

    if isinstance(data, dict):
        data = [data]
    
    candidates = [entry.get(key_name) for entry in data if isinstance(entry, dict)]
    candidate_id = new_data.get(key_name)
    
    if candidate_id in candidates:
        print(f"WARNING: Resume '{candidate_id}' already registered. Ignoring addition.")
        return

    data.append(new_data)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)    
    
    print(f"✅ Resume '{candidate_id}' successfully saved.")

def display_json_table(path_json: str) -> pd.DataFrame:
    """JSON dosyasını okur ve Pandas DataFrame olarak döndürür (Güvenilir)."""
    data = load_json_cv(path_json)
    if not data:
        return pd.DataFrame()
        
    # İyileştirme: pd.json_normalize, iç içe geçmiş listeleri düzgün işler.
    df = pd.json_normalize(data) 
    return df
    
# --- ANA İŞ AKIŞI FONKSİYONU ---

def process_cv_analysis(schema, job_details, prompt_template, prompt_score, llm, file_path):
    """
    CV dosyasını işler, LLM'e gönderir, JSON'a ayrıştırır ve yapılandırılmış sonucu döndürür.
    """
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"File not found: {file_path}")

    content = parse_document_to_markdown(file_path) 

    chain = prompt_template | llm
    
    output = chain.invoke({
        "schema": schema, 
        "resume": content, 
        "job": job_details, 
        "prompt_score": prompt_score
    })
    
    raw_response_text = output.content
    
    try:
        # Onarım ve formatlama
        structured_data = format_result(raw_response_text)
    except Exception as e:
        # Kritik JSON ayrıştırma hatası fırlat
        raise json.JSONDecodeError(f"Kritik JSON Ayrıştırma Hatası: Model çıktısı bozuk. Detay: {e}", raw_response_text, 0)

    return structured_data