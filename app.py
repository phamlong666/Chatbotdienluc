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

# --- THƯ VIỆN CHO SEMANTIC SEARCH ---
from sentence_transformers import SentenceTransformer, util
import torch

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

# Cấu hình Matplotlib hiển thị tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

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

# 3. KHỞI TẠO CÔNG CỤ AI
@st.cache_resource
def init_ai_tools():
    return {
        "semantic_model": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
        "ocr_reader": easyocr.Reader(['vi', 'en'])
    }

ai_tools = init_ai_tools()

# 4. KẾT NỐI GOOGLE SHEETS
def get_sheets_connection():
    if secrets_data["gdrive"]:
        try:
            creds = Credentials.from_service_account_info(secrets_data["gdrive"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
            return gspread.authorize(creds)
        except: return None
    return None

gc = get_sheets_connection()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

# 5. CÁC HÀM XỬ LÝ DỮ LIỆU LOGIC CŨ
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=300)
def load_all_sheets():
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sh.worksheets()}
    except: return {}

def extract_year_month(text):
    months = re.findall(r'tháng\s+(\d+)', text)
    years = re.findall(r'năm\s+(\d{4})', text)
    m = int(months[0]) if months else None
    y = int(years[0]) if years else None
    return m, y

# 6. GIAO DIỆN CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=100)
    st.title("AI Đội Định Hóa")
    if st.button("🗑 Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()

# HIỂN THỊ CHAT
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg: st.dataframe(msg["df"])
        if "fig" in msg: st.pyplot(msg["fig"])

u_input = st.chat_input("Hỏi về KPI, CBCNV, Lãnh đạo xã hoặc Sự cố...")

if u_input:
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"): st.markdown(u_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang lục tìm dữ liệu..."):
            all_data = load_all_sheets()
            handled = False
            norm_u = normalize_text(u_input)
            month, year = extract_year_month(norm_u)

            # --- LUỒNG 1: DỮ LIỆU KPI ---
            if any(k in norm_u for k in ["kpi", "chỉ số", "thực hiện"]):
                df = all_data.get("KPI", pd.DataFrame())
                if not df.empty:
                    # Logic lọc theo tháng/năm nếu có trong câu hỏi
                    if month: df = df[df['Tháng'] == month]
                    if year: df = df[df['Năm'] == year]
                    
                    res = f"Đây là dữ liệu KPI {f'tháng {month}' if month else ''} {f'năm {year}' if year else ''} ạ:"
                    st.markdown(res)
                    st.dataframe(df)
                    
                    # Tự động vẽ biểu đồ nếu yêu cầu
                    if "biểu đồ" in norm_u:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(data=df, x='Đơn vị', y='Tỷ lệ thực hiện', ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df, "fig": fig})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df})
                    handled = True

            # --- LUỒNG 2: CBCNV ---
            elif any(k in norm_u for k in ["cbcnv", "nhân viên", "nhân sự"]):
                df = all_data.get("CBCNV", pd.DataFrame())
                if not df.empty:
                    res = f"Đội hiện có {len(df)} CBCNV. Danh sách chi tiết:"
                    st.markdown(res)
                    st.dataframe(df)
                    
                    if "biểu đồ" in norm_u:
                        fig, ax = plt.subplots()
                        df['Trình độ'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        st.pyplot(fig)
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df, "fig": fig})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df})
                    handled = True

            # --- LUỒNG 3: LÃNH ĐẠO XÃ ---
            elif any(k in norm_u for k in ["lãnh đạo", "xã", "địa phương"]):
                df = all_data.get("Lãnh đạo xã", pd.DataFrame())
                if not df.empty:
                    # Lọc theo tên xã nếu có
                    for x in df['Xã'].unique():
                        if normalize_text(x) in norm_u:
                            df = df[df['Xã'] == x]
                            break
                    st.markdown("Thông tin lãnh đạo địa phương:")
                    st.dataframe(df)
                    st.session_state.messages.append({"role": "assistant", "content": "Dữ liệu lãnh đạo xã ạ.", "df": df})
                    handled = True

            # --- LUỒNG 4: SỰ CỐ ---
            elif any(k in norm_u for k in ["sự cố", "mất điện"]):
                df = all_data.get("Sự cố", pd.DataFrame())
                if not df.empty:
                    if year: df = df[df['Năm'] == year]
                    res = "Thống kê tình hình sự cố:"
                    st.markdown(res)
                    st.dataframe(df)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df})
                    handled = True

            # --- LUỒNG 5: HỎI ĐÁP THÔNG MINH ---
            if not handled:
                df_qa = all_data.get("Hỏi-Trả lời", pd.DataFrame())
                if not df_qa.empty:
                    qs = df_qa["Câu hỏi"].tolist()
                    q_embs = ai_tools["semantic_model"].encode(qs, convert_to_tensor=True)
                    u_emb = ai_tools["semantic_model"].encode(u_input, convert_to_tensor=True)
                    hits = util.cos_sim(u_emb, q_embs)[0]
                    best = torch.argmax(hits).item()
                    if hits[best] > 0.4:
                        ans = df_qa.iloc[best]["Câu trả lời"]
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                        handled = True

            # --- LUỒNG 6: GEMINI AI ---
            if not handled and secrets_data["gemini"]:
                try:
                    genai.configure(api_key=secrets_data["gemini"])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Bạn là trợ lý Đội Định Hóa. Anh Long hỏi: {u_input}")
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    handled = True
                except: pass

            if not handled:
                err = "Em chưa thấy thông tin này trong dữ liệu ạ."
                st.info(err)
                st.session_state.messages.append({"role": "assistant", "content": err})