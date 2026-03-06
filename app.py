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
from fuzzywuzzy import fuzz
import io
from PIL import Image

# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN CHUYÊN NGHIỆP (PHONG CÁCH GEMINI)
st.set_page_config(layout="wide", page_title="Đội Định Hóa AI", page_icon="✨")

# Tùy chỉnh giao diện bằng CSS để chia 2 bên trái/phải
st.markdown("""
    <style>
    /* Tổng thể */
    .stApp {
        background-color: #ffffff;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }
    
    /* Ép tin nhắn hiển thị 2 bên */
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
        display: flex !important;
        width: 100% !important;
    }

    /* Tin nhắn NGƯỜI DÙNG (Bên phải) */
    [data-testid="stChatMessageUser"] {
        background-color: #f0f4f9 !important;
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        max-width: 80% !important;
        border-bottom-right-radius: 4px;
    }

    /* Tin nhắn CHATBOT (Bên trái) */
    [data-testid="stChatMessageAssistant"] {
        background-color: #ffffff !important;
        border: 1px solid #e3e3e3;
        margin-right: auto !important;
        max-width: 85% !important;
        border-bottom-left-radius: 4px;
    }

    /* Tinh chỉnh Avatar */
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageAvatar"] {
        margin-left: 10px;
        margin-right: 0;
    }

    /* Thanh input */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    h1, h2, h3 {
        color: #1f1f1f;
        font-family: 'Google Sans', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. GIẢI MÃ KHÓA BẢO MẬT
@st.cache_resource
def get_decrypted_all_keys():
    config = {"gemini": None, "gdrive": None}
    if "gdrive_service_account" in st.secrets:
        try:
            sec = st.secrets["gdrive_service_account"]
            master_key = sec.get("encryption_key_for_decryption").encode()
            cipher = Fernet(master_key)
            
            # Giải mã Gemini API Key
            enc_gemini = sec.get("encrypted_gemini_api_key")
            if enc_gemini:
                config["gemini"] = cipher.decrypt(enc_gemini.encode()).decode()
            
            # Giải mã Google Drive Private Key
            enc_g_private = sec.get("encrypted_private_key").encode()
            dec_g_private = cipher.decrypt(enc_g_private).decode()
            config["gdrive"] = {
                "type": sec.get("type", "service_account"),
                "project_id": sec.get("project_id"),
                "private_key_id": sec.get("private_key_id"),
                "private_key": dec_g_private,
                "client_email": sec.get("client_email"),
                "client_id": sec.get("client_id"),
                "auth_uri": sec.get("auth_uri"),
                "token_uri": sec.get("token_uri"),
                "auth_provider_x509_cert_url": sec.get("auth_provider_x509_cert_url"),
                "client_x509_cert_url": sec.get("client_x509_cert_url")
            }
        except Exception as e:
            st.error(f"Lỗi giải mã: {e}")
    return config

secrets_data = get_decrypted_all_keys()

# 3. KẾT NỐI GOOGLE SHEETS
def get_sheets_connection():
    if secrets_data["gdrive"]:
        try:
            creds = Credentials.from_service_account_info(secrets_data["gdrive"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
            return gspread.authorize(creds)
        except: return None
    return None

gc = get_sheets_connection()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

# 4. CÔNG CỤ XỬ LÝ DỮ LIỆU
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=60) # Giảm TTL để cập nhật sheet nhanh hơn
def load_data_from_sheets():
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        data = {}
        for ws in sh.worksheets():
            rows = ws.get_all_records()
            if rows:
                df = pd.DataFrame(rows)
                # Làm sạch tên cột (xóa khoảng trắng thừa)
                df.columns = [str(c).strip() for c in df.columns]
                data[ws.title] = df
        return data
    except Exception as e:
        return {}

def query_gemini(prompt, files=None):
    if not secrets_data["gemini"]: return "⚠️ Chưa cấu hình API Gemini."
    try:
        genai.configure(api_key=secrets_data["gemini"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        content = [prompt]
        if files: content.extend(files)
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# 5. GIAO DIỆN CHÍNH
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=120)
    st.markdown("### ✨ Đội Định Hóa AI")
    st.caption("Trợ lý phân tích dữ liệu & Nghiệp vụ")
    
    st.divider()
    st.markdown("### 📄 Phân tích tài liệu")
    uploaded_files = st.file_uploader("Tải Word, Excel, PDF, Ảnh", 
                                    type=["pdf", "xlsx", "docx", "png", "jpg"], 
                                    accept_multiple_files=True)
    
    if st.button("🗑 Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg: st.dataframe(msg["df"])

# Input người dùng
u_input = st.chat_input("Hỏi em bất cứ điều gì...")

if u_input:
    # 1. Hiển thị câu hỏi người dùng (Bên phải)
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"):
        st.markdown(u_input)

    # 2. Phản hồi của Chatbot (Bên trái)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm dữ liệu..."):
            all_data = load_data_from_sheets()
            handled = False
            
            # --- LUỒNG 1: TRA CỨU TRONG HỎI-TRẢ LỜI (DỮ LIỆU G-SHEET) ---
            # Tìm trong sheet "Hỏi-Trả lời" hoặc "sample_questions"
            df_qa = all_data.get("Hỏi-Trả lời", all_data.get("sample_questions", pd.DataFrame()))
            
            if not df_qa.empty and ("Câu hỏi" in df_qa.columns):
                best_ans = None
                highest_score = 0
                norm_input = normalize_text(u_input)
                
                for _, row in df_qa.iterrows():
                    q_in_sheet = str(row.get("Câu hỏi", ""))
                    score = fuzz.token_set_ratio(norm_input, normalize_text(q_in_sheet))
                    
                    if score > 80: # Ngưỡng khớp cao
                        best_ans = str(row.get("Câu trả lời", ""))
                        highest_score = score
                        break # Tìm thấy là dừng luôn
                
                if best_ans:
                    st.markdown(best_ans)
                    st.session_state.messages.append({"role": "assistant", "content": best_ans})
                    handled = True

            # --- LUỒNG 2: PHÂN TÍCH FILE NẾU CÓ ---
            if not handled and uploaded_files:
                ans = query_gemini(f"Dựa trên các file đã tải lên, hãy trả lời câu hỏi: {u_input}")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                handled = True

            # --- LUỒNG 3: TRA CỨU KPI / CBCNV (DỮ LIỆU CẤU TRÚC) ---
            if not handled:
                norm_u = normalize_text(u_input)
                if any(k in norm_u for k in ["kpi", "chỉ số"]):
                    df_kpi = all_data.get("KPI", pd.DataFrame())
                    if not df_kpi.empty:
                        res = "Đây là bảng KPI em tìm thấy:"
                        st.markdown(res)
                        st.dataframe(df_kpi)
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df_kpi})
                        handled = True
                elif any(k in norm_u for k in ["nhân viên", "nhân sự", "cbcnv"]):
                    df_nv = all_data.get("CBCNV", pd.DataFrame())
                    if not df_nv.empty:
                        res = "Danh sách CBCNV Đội Định Hóa:"
                        st.markdown(res)
                        st.dataframe(df_nv)
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df_nv})
                        handled = True

            # --- LUỒNG 4: GEMINI AI (CÂU HỎI TỰ DO) ---
            if not handled:
                ans = query_gemini(f"Bạn là trợ lý Đội Định Hóa. Hãy trả lời: {u_input}")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})