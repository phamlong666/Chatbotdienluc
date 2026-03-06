# -*- coding: utf-8 -*-
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import unicodedata
import datetime
import easyocr
import speech_recognition as sr
import tempfile
import numpy as np
from cryptography.fernet import Fernet
from audio_recorder_streamlit import audio_recorder
import seaborn as sns
import google.generativeai as genai

# --- THƯ VIỆN CHO SEMANTIC SEARCH (TÌM KIẾM THEO Ý NGHĨA) ---
from sentence_transformers import SentenceTransformer, util
import torch

# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN CHUNG
st.set_page_config(layout="wide", page_title="AI Điện lực Định Hóa", page_icon="🤖")

# Cấu hình hiển thị tiếng Việt cho biểu đồ Matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False 

# 2. HÀM GIẢI MÃ BẢO MẬT (DÙNG CHUNG CHO GDRIVE VÀ GEMINI)
@st.cache_resource
def get_decrypted_all_keys():
    """Giải mã các khóa bí mật được lưu trong Streamlit Secrets"""
    config = {"gemini": None, "gdrive": None}
    
    if "gdrive_service_account" in st.secrets:
        try:
            sec = st.secrets["gdrive_service_account"]
            # Khóa chính dùng để giải mã (Master Key)
            master_key = sec.get("encryption_key_for_decryption").encode()
            cipher = Fernet(master_key)
            
            # 2.1 Giải mã khóa Gemini API (Lấy từ Google AI Studio)
            enc_gemini = sec.get("encrypted_gemini_api_key")
            if enc_gemini:
                config["gemini"] = cipher.decrypt(enc_gemini.encode()).decode()
            
            # 2.2 Giải mã khóa Private Key của Google Service Account
            enc_g_private = sec.get("encrypted_private_key").encode()
            dec_g_private = cipher.decrypt(enc_g_private).decode()
            
            config["gdrive"] = {
                "type": sec.get("type", "service_account"),
                "project_id": sec.get("project_id"),
                "private_key_id": sec.get("private_key_id"),
                "private_key": dec_g_private,
                "client_email": sec.get("client_email"),
                "client_id": sec.get("client_id"),
                "auth_uri": sec.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": sec.get("token_uri", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": sec.get("auth_provider_x509_cert_url"),
                "client_x509_cert_url": sec.get("client_x509_cert_url")
            }
        except Exception as e:
            st.error(f"Lỗi hệ thống bảo mật: {e}")
            
    return config

secrets_data = get_decrypted_all_keys()

# 3. KHỞI TẠO CÁC MÔ HÌNH AI (CACHE ĐỂ TIẾT KIỆM RAM)
@st.cache_resource
def init_ai_tools():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "semantic_model": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device),
        "ocr_reader": easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
    }

with st.spinner("🤖 Đang khởi động hệ thống trợ lý..."):
    ai_tools = init_ai_tools()

# 4. KẾT NỐI VỚI DỮ LIỆU GOOGLE SHEETS
def get_sheets_connection():
    if secrets_data["gdrive"]:
        try:
            creds = Credentials.from_service_account_info(
                secrets_data["gdrive"], 
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            return gspread.authorize(creds)
        except Exception as e:
            st.error(f"Lỗi kết nối Google Sheets: {e}")
    return None

gc = get_sheets_connection()
# Đường dẫn file Google Sheets của anh Long
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

# 5. CÁC HÀM XỬ LÝ DỮ LIỆU
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=600)
def load_all_sheets():
    """Tải toàn bộ dữ liệu từ các sheet vào DataFrame"""
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sh.worksheets()}
    except Exception as e:
        return {}

def semantic_search(query, df, q_col, a_col, threshold=0.45):
    """Tìm kiếm câu trả lời dựa trên sự tương đồng về ý nghĩa (Vector Search)"""
    if df.empty or q_col not in df.columns: return None, 0
    questions = df[q_col].astype(str).tolist()
    doc_embs = ai_tools["semantic_model"].encode(questions, convert_to_tensor=True)
    query_emb = ai_tools["semantic_model"].encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, doc_embs)[0]
    best_idx = torch.argmax(cos_scores).item()
    score = float(cos_scores[best_idx])
    
    if score > threshold:
        return df.iloc[best_idx][a_col], score * 100
    return None, 0

# 6. GIAO DIỆN NGƯỜI DÙNG Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_or_ocr_text" not in st.session_state:
    st.session_state.voice_or_ocr_text = None

# Sidebar chứa các công cụ
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=100)
    st.title("Trợ lý Điện lực Định Hóa")
    
    st.divider()
    st.subheader("🎙 Nhập bằng giọng nói")
    audio_val = audio_recorder(text="Bấm để nói", icon_size="2x", key="voice_rec")
    if audio_val:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_val)
            tmp_path = tmp.name
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
            try:
                st.session_state.voice_or_ocr_text = r.recognize_google(audio, language="vi-VN")
            except: st.error("Không rõ âm thanh...")
        os.remove(tmp_path)

    st.subheader("📷 Quét ảnh (OCR)")
    pic = st.file_uploader("Tải ảnh văn bản/bảng biểu", type=['jpg', 'png', 'jpeg'])
    if pic:
        with st.spinner("Đang trích xuất chữ..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(pic.read())
                tmp_path = tmp.name
            ocr_res = ai_tools["ocr_reader"].readtext(tmp_path, detail=0)
            st.session_state.voice_or_ocr_text = " ".join(ocr_res)
            os.remove(tmp_path)

    if st.button("🗑 Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()

# Hiển thị hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fig" in msg: st.pyplot(msg["fig"])
        if "df" in msg: st.dataframe(msg["df"])

# Xử lý Input (từ Chatbox hoặc Giọng nói/OCR)
u_input = st.session_state.voice_or_ocr_text if st.session_state.voice_or_ocr_text else st.chat_input("Hỏi về TBA, KPI, CBCNV...")
st.session_state.voice_or_ocr_text = None

if u_input:
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"): st.markdown(u_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm dữ liệu..."):
            all_data = load_all_sheets()
            handled = False
            norm_u = normalize_text(u_input)

            # LỚP 1: DỮ LIỆU NỘI BỘ GOOGLE SHEETS (QA)
            df_qa = all_data.get("Hỏi-Trả lời", pd.DataFrame())
            ans, sc = semantic_search(u_input, df_qa, "Câu hỏi", "Câu trả lời")
            if ans:
                res = f"📌 **Thông tin nội bộ ({sc:.0f}% khớp):**\n\n{ans}"
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
                handled = True

            # LỚP 2: NGHIỆP VỤ BIỂU ĐỒ & NHÂN SỰ
            if not handled:
                # 2.1 KPI & Biểu đồ
                if any(k in norm_u for k in ["kpi", "biểu đồ", "thống kê"]):
                    df_kpi = all_data.get("KPI", pd.DataFrame())
                    if not df_kpi.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(data=df_kpi.head(15), x=df_kpi.columns[0], y=df_kpi.columns[1], ax=ax, palette="mako")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        txt = "Đây là biểu đồ KPI Điện lực mình anh Long nhé."
                        st.markdown(txt)
                        st.session_state.messages.append({"role": "assistant", "content": txt, "fig": fig})
                        handled = True
                
                # 2.2 Danh sách nhân sự CBCNV
                elif any(k in norm_u for k in ["cbcnv", "nhân viên", "danh sách"]):
                    df_cb = all_data.get("CBCNV", pd.DataFrame())
                    if not df_cb.empty:
                        st.dataframe(df_cb, use_container_width=True)
                        txt = f"Dạ, danh sách hiện có {len(df_cb)} cán bộ công nhân viên ạ."
                        st.markdown(txt)
                        st.session_state.messages.append({"role": "assistant", "content": txt, "df": df_cb})
                        handled = True

            # LỚP 3: GOOGLE GEMINI AI (XỬ LÝ KIẾN THỨC CHUNG)
            if not handled and secrets_data["gemini"]:
                try:
                    genai.configure(api_key=secrets_data["gemini"])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    # Prompt chuyên biệt cho Điện lực Định Hóa
                    ctx = f"Bạn là trợ lý ảo Điện lực Định Hóa. Hãy trả lời anh Long một cách thân thiện: {u_input}"
                    response = model.generate_content(ctx)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    handled = True
                except Exception as e:
                    st.warning("🤖 Hiện tại AI đang bận, anh vui lòng thử lại sau giây lát.")

            if not handled:
                st.info("Em chưa tìm thấy thông tin này trong dữ liệu nội bộ. Anh cần em giúp gì thêm không ạ?")