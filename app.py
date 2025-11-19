import streamlit as st
import os
import sys # <-- Import sys
import json
import pandas as pd
from dotenv import load_dotenv

# --- CRITICAL FIX: Add the 'src' directory to the Python path ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# Only append if not already in path (for safety)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# --- END CRITICAL FIX ---


# 1. Import helper and prompt as if they were top-level modules
#    (Because we added the parent 'src' directory to the path)
from helper import (
    load_local_llm,
    load_gemini_llm, 
    load_job,
    save_json_cv,
    process_cv_analysis
)
from prompt import (
    prompt_template, 
    schema, 
    prompt_score
)

# --- KONFİGÜRASYON VE HAZIRLIK ---

load_dotenv()
# Dosya yollarını projenize göre ayarlayın
# NOT: Bu yolların uygulamanın kök dizininde çalıştığından emin olun
JOB_CSV_PATH = "result/jobs.csv" 
RESUME_JSON_PATH = "result/resumes.json" 

st.set_page_config(page_title="LLM CV Analyzer", layout="wide")
st.title("🤖 LLM Destekli Özgeçmiş Analiz Aracı")
st.markdown("---")


# --- HELPER FONKSİYON: RAPOR GÖRÜNTÜLEME (Streamlit Uyumlu) ---

def generate_report_markdown(result):
    """Analiz sonuçlarını Markdown metni olarak hazırlar."""
    
    candidate_name = result.get('name', 'Candidate Name Not Found')
    md = f"## 📄 Candidate Resume Analysis Report: {candidate_name}\n"
    md += f"--- \n"
    
    # 1. Core Data
    md += f"### 1. Core Profile and Background\n"
    md += f"| Criterion | Value |\n"
    md += f"| :--- | :--- |\n"
    md += f"| **Candidate Name** | **{candidate_name}** |\n"
    md += f"| **Primary Area** | {result.get('area', 'N/A')} |\n"
    md += f"| **Education Summary** | {result.get('education', 'Not specified')} |\n"
    md += "\n"
    
    # 2. Executive Summary
    md += f"### 2. Executive Summary\n"
    md += f"> {result.get('summary', 'Summary not extracted.')}\n\n"
    
    # 3. Technical Skills
    md += f"### 3. Technical and Core Skills\n"
    skills = result.get('skills', [])
    if isinstance(skills, list) and skills:
        md += f"**Key Skills:** {', '.join([f'**{s}**' for s in skills])}\n\n"
    else:
        md += "* Skills section was not clearly structured or is empty.\n\n"

    # ... (Diğer tüm bölümlerin Markdown oluşturma mantığı buraya eklenir)
    
    # 5. Fit & Development Analysis
    md += f"### 5. Fit & Development Analysis\n"
    
    # Strengths
    md += f"#### 5.1. Core Strengths\n"
    strengths = result.get('strengths', [])
    if isinstance(strengths, list) and strengths:
        for item in strengths:
            md += f"* **✅ {item}**\n"
    
    # ... (Gaps, Considerations, Recommendation mantığı buraya eklenir)
    
    return md # Final Markdown string'i döndürülür.


# --- SOL SÜTUN: KONTROLLER ---
with st.sidebar:
    st.header("⚙️ Ayarlar ve Veri Girişi")

    # LLM Seçimi için st.radio kullanımı
    model_choice = st.radio(
        "LLM Kaynağını Seçin:",
        ('Yerel (Ollama: llama3)', 'Bulut (Gemini API)')
    )
    
    # --- Model Yükleme ---
    llm = None
    if model_choice == 'Yerel (Ollama: llama3)':
        llm = load_local_llm(model="llama3", temperature=0)
        st.success("Yerel Llama3 modeli yüklendi.")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("⚠️ GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")
            # llm None kalır
        else:
            llm = load_gemini_llm(id_model="gemini-2.5-flash", temperature=0)
            st.success("Bulut Gemini modeli yüklendi.")

    st.markdown("---")
    
    # --- İş Tanımını Yükleme ---
    job_details_text = load_job(JOB_CSV_PATH)
    if "Error" in job_details_text:
        st.error(f"İş Tanımı CSV Yükleme Hatası: {job_details_text}")
    else:
        st.success("İş Tanımı Başarıyla Yüklendi.")
        st.markdown("**Analiz Edilen Pozisyon:**")
        st.markdown(job_details_text)

    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "PDF Özgeçmiş Dosyasını Yükle", 
        type=["pdf"]
    )


# --- ANA PENCERE: ANALİZ VE GÖRÜNTÜLEME ---
if uploaded_file is not None and llm is not None:
    # Yüklenen dosyayı geçici olarak kaydet
    temp_file_path = os.path.join("./temp_cv.pdf")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Analizi Başlat", key="analyze_btn"):
        with st.spinner("LLM özgeçmişi analiz ediyor ve puanlıyor... Lütfen bekleyin."):
            try:
                # ANA ANALİZ ÇAĞRISI
                structured_data = process_cv_analysis(
                    schema, 
                    job_details_text, 
                    prompt_template, 
                    prompt_score, 
                    llm, 
                    temp_file_path
                )
                
                # Sonuçları kaydet
                save_json_cv(structured_data, RESUME_JSON_PATH)

                st.success("Analiz Tamamlandı!")
                st.balloons()
                
                # Raporu göster
                st.markdown(generate_report_markdown(structured_data))

            except Exception as e:
                st.error(f"Analiz sırasında kritik hata oluştu.")
                st.exception(e)
                
    # Geçici dosyayı temizle
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

elif llm is None:
    st.error("LLM Yüklenemedi. Lütfen bir model seçimi yapın ve API anahtarınızı kontrol edin.")
else:
    st.info("Lütfen sol panelden bir CV dosyası yükleyin ve analizi başlatın.")