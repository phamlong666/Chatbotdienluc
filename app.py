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

# Sửa lỗi Import: Sử dụng cấu trúc an toàn hơn
try:
    from fuzzywuzzy import fuzz
except ImportError:
    st.error("Thiếu thư viện fuzzywuzzy. Anh vui lòng thêm 'fuzzywuzzy' và 'python-Levenshtein' vào requirements.txt trên GitHub.")

# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN CHUNG
st.set_page_config(layout="wide", page_title="AI Đội Định Hóa", page_icon="🤖")

# CSS Tùy chỉnh giao diện Chat chuẩn
st.markdown("""
    <style>
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex !important;
        width: fit-content !important;
        max-width: 85% !important;
    }
    [data-testid="stChatMessageUser"] {
        background-color: #e3f2fd !important;
        margin-left: auto !important;
        flex-direction: row-reverse !important;
        text-align: right;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #f5f5f5 !important;
        margin-right: auto !important;
    }
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageAvatar"] {
        margin-left: 10px; margin-right: 0;
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
            enc_gemini = sec.get("encrypted_gemini_api_key")
            if enc_gemini:
                config["gemini"] = cipher.decrypt(enc_gemini.encode()).decode()
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
            st.error(f"Lỗi giải mã bảo mật: {e}")
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

# 4. CÁC HÀM XỬ LÝ DỮ LIỆU
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=300)
def load_all_sheets():
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        data = {}
        for ws in sh.worksheets():
            rows = ws.get_all_records()
            if rows:
                df = pd.DataFrame(rows)
                df.columns = [str(c).strip() for c in df.columns]
                data[ws.title] = df
        return data
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return {}

def find_best_answer(user_query, df_qa):
    if df_qa.empty or "Câu hỏi" not in df_qa.columns:
        return None, 0
    
    best_answer = None
    max_score = 0
    norm_query = normalize_text(user_query)
    
    for _, row in df_qa.iterrows():
        question = str(row["Câu hỏi"])
        answer = str(row["Câu trả lời"])
        # Logic Fuzzy Matching tương đương app-001
        score = fuzz.token_set_ratio(norm_query, normalize_text(question))
        
        if score > max_score:
            max_score = score
            best_answer = answer
            
    return best_answer, max_score

# 5. GIAO DIỆN CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=100)
    st.title("AI Đội Định Hóa")
    if st.button("🗑 Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg: st.dataframe(msg["df"])

u_input = st.chat_input("Hỏi em bất cứ điều gì...")

if u_input:
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"): st.markdown(u_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang kiểm tra dữ liệu..."):
            all_data = load_all_sheets()
            handled = False
            norm_u = normalize_text(u_input)

            # --- LUỒNG 1: KIỂM TRA TRONG HỎI - TRẢ LỜI ---
            df_qa = all_data.get("Hỏi-Trả lời", pd.DataFrame())
            ans, score = find_best_answer(u_input, df_qa)
            
            if ans and score > 65:
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                handled = True

            # --- LUỒNG 2: KIỂM TRA DỮ LIỆU CÔNG VIỆC ---
            if not handled:
                if any(k in norm_u for k in ["kpi", "chỉ số"]):
                    df = all_data.get("KPI", pd.DataFrame())
                    if not df.empty:
                        st.markdown("Dữ liệu KPI Đội Định Hóa:")
                        st.dataframe(df)
                        st.session_state.messages.append({"role": "assistant", "content": "Bảng KPI đây ạ:", "df": df})
                        handled = True
                
                elif any(k in norm_u for k in ["nhân viên", "nhân sự", "cbcnv"]):
                    df = all_data.get("CBCNV", pd.DataFrame())
                    if not df.empty:
                        st.markdown(f"Danh sách {len(df)} CBCNV:")
                        st.dataframe(df)
                        st.session_state.messages.append({"role": "assistant", "content": "Danh sách nhân sự đây ạ:", "df": df})
                        handled = True

            # --- LUỒNG 3: GEMINI (NẾU SHEET KHÔNG CÓ) ---
            if not handled and secrets_data["gemini"]:
                try:
                    genai.configure(api_key=secrets_data["gemini"])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Bạn là trợ lý Đội Định Hóa. Hãy trả lời anh Long: {u_input}")
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    handled = True
                except:
                    st.error("AI Gemini đang bận.")

            if not handled:
                msg = "Dạ em chưa tìm thấy thông tin này trong tài liệu nội bộ."
                st.info(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})