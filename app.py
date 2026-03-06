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
st.set_page_config(layout="wide", page_title="AI Đội Định Hóa", page_icon="🤖")

# CSS để chia tin nhắn sang 2 bên
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    /* Tin nhắn người dùng (phải) */
    [data-testid="stChatMessage"]:nth-child(even) {
        margin-left: auto;
        background-color: #e3f2fd;
    }
    /* Tin nhắn bot (trái) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        margin-right: auto;
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# Cấu hình Matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False 

# 2. HÀM GIẢI MÃ BẢO MẬT
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
                "auth_uri": sec.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": sec.get("token_uri", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": sec.get("auth_provider_x509_cert_url"),
                "client_x509_cert_url": sec.get("client_x509_cert_url")
            }
        except Exception: pass
    return config

secrets_data = get_decrypted_all_keys()

# 3. KHỞI TẠO MÔ HÌNH AI
@st.cache_resource
def init_ai_tools():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "semantic_model": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device),
        "ocr_reader": easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
    }

with st.spinner("🤖 Đang kết nối trí tuệ nhân tạo..."):
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

# 5. XỬ LÝ DỮ LIỆU
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=600)
def load_all_sheets():
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sh.worksheets()}
    except: return {}

def semantic_search(query, df, q_col, a_col, threshold=0.5):
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

# 6. GIAO DIỆN
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_or_ocr_text" not in st.session_state:
    st.session_state.voice_or_ocr_text = None

with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=100)
    st.title("Trợ lý Đội Định Hóa")
    st.divider()
    audio_val = audio_recorder(text="Bấm để nói", icon_size="2x")
    if audio_val:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_val)
            tmp_path = tmp.name
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            try:
                audio = r.record(source)
                st.session_state.voice_or_ocr_text = r.recognize_google(audio, language="vi-VN")
            except: st.error("Lỗi âm thanh...")
        os.remove(tmp_path)

    if st.button("🗑 Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()

# Hiển thị hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fig" in msg: st.pyplot(msg["fig"])
        if "df" in msg: st.dataframe(msg["df"])

u_input = st.session_state.voice_or_ocr_text if st.session_state.voice_or_ocr_text else st.chat_input("Nhập câu hỏi tại đây...")
st.session_state.voice_or_ocr_text = None

if u_input:
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"): st.markdown(u_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất dữ liệu..."):
            all_data = load_all_sheets()
            handled = False
            norm_u = normalize_text(u_input)

            # --- LỚP 1: TÌM KIẾM THEO TỪ KHÓA Ý ĐỊNH (Intent) ---
            
            # 1.1 KPI
            if any(k in norm_u for k in ["kpi", "chỉ số"]):
                df_kpi = all_data.get("KPI", pd.DataFrame())
                if not df_kpi.empty:
                    # Logic sắp xếp nếu có yêu cầu
                    if "giảm dần" in norm_u:
                        df_kpi = df_kpi.sort_values(by=df_kpi.columns[1], ascending=False)
                    
                    st.dataframe(df_kpi)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(data=df_kpi.head(10), x=df_kpi.columns[0], y=df_kpi.columns[1], palette="Blues_d")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    res = "Dạ đây là dữ liệu KPI anh cần ạ."
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig, "df": df_kpi})
                    handled = True

            # 1.2 CBCNV & BIỂU ĐỒ TUỔI/TRÌNH ĐỘ
            elif any(k in norm_u for k in ["cbcnv", "nhân viên", "độ tuổi", "trình độ"]):
                df_cb = all_data.get("CBCNV", pd.DataFrame())
                if not df_cb.empty:
                    if "độ tuổi" in norm_u:
                        fig, ax = plt.subplots()
                        df_cb.iloc[:, 1].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        st.pyplot(fig)
                        res = "Đây là biểu đồ cơ cấu độ tuổi CBCNV."
                        st.markdown(res)
                        st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig})
                    else:
                        st.dataframe(df_cb)
                        res = f"Danh sách CBCNV hiện có {len(df_cb)} người."
                        st.markdown(res)
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df_cb})
                    handled = True

            # 1.3 LÃNH ĐẠO XÃ
            elif "lãnh đạo xã" in norm_u:
                df_ld = all_data.get("Lãnh đạo xã", pd.DataFrame())
                if not df_ld.empty:
                    # Tìm xã cụ thể
                    for xa in ["định hóa", "kim phượng", "phượng tiến", "trung hội", "bình yên", "phú đình"]:
                        if xa in norm_u:
                            df_filtered = df_ld[df_ld.iloc[:, 0].str.lower().contains(xa)]
                            st.table(df_filtered)
                            res = f"Thông tin lãnh đạo xã {xa.title()} đây ạ."
                            st.markdown(res)
                            st.session_state.messages.append({"role": "assistant", "content": res, "df": df_filtered})
                            handled = True
                            break

            # 1.4 TBA & ĐƯỜNG DÂY
            elif any(k in norm_u for k in ["tba", "trạm biến áp", "đường dây"]):
                df_tba = all_data.get("TBA", pd.DataFrame())
                if not df_tba.empty:
                    res = "Thông tin trạm biến áp chi tiết:"
                    st.dataframe(df_tba)
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df_tba})
                    handled = True

            # --- LỚP 2: TÌM TRONG HỎI - TRẢ LỜI (DÀNH CHO CÂU HỎI CỐ ĐỊNH) ---
            if not handled:
                ans, sc = semantic_search(u_input, all_data.get("Hỏi-Trả lời", pd.DataFrame()), "Câu hỏi", "Câu trả lời")
                if ans:
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    handled = True

            # --- LỚP 3: GOOGLE GEMINI (DÀNH CHO CÂU HỎI TỰ DO) ---
            if not handled and secrets_data["gemini"]:
                try:
                    genai.configure(api_key=secrets_data["gemini"])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Bạn là trợ lý Đội Định Hóa. Hãy trả lời anh Long: {u_input}")
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    handled = True
                except: st.warning("Hệ thống AI đang bận...")

            if not handled:
                st.info("Em chưa tìm thấy dữ liệu này. Anh có thể thử lại với từ khóa khác như 'KPI', 'Danh sách nhân viên'...")