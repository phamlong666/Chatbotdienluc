# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# =========================================
# CẤU HÌNH GIAO DIỆN
# =========================================

st.set_page_config(
    page_title="Chatbot Điện lực",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Chatbot Phân tích dữ liệu Điện lực")
st.write("Xin chào! Tôi là trợ lý AI giúp bạn phân tích dữ liệu Google Sheets.")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("📊 Hệ thống Chatbot")
st.sidebar.write("Ứng dụng AI phân tích dữ liệu nội bộ")

# =========================================
# GOOGLE SHEETS
# =========================================

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

info = dict(st.secrets["gdrive_service_account"])

creds = Credentials.from_service_account_info(
    info,
    scopes=SCOPES
)

client = gspread.authorize(creds)

spreadsheet = client.open_by_url(
    st.secrets["spreadsheet_url"]
)

# =========================================
# LOAD DATA
# =========================================

@st.cache_data
def load_data():

    data = {}

    for ws in spreadsheet.worksheets():

        df = pd.DataFrame(ws.get_all_records())

        data[ws.title] = df

    return data


data = load_data()

# =========================================
# GEMINI AI
# =========================================

genai.configure(api_key=st.secrets["gemini_api_key"])

model = genai.GenerativeModel("gemini-1.5-flash")

# =========================================
# HIỂN THỊ DATA
# =========================================

with st.expander("📑 Xem dữ liệu Google Sheets"):

    for name, df in data.items():

        st.subheader(name)
        st.dataframe(df)

# =========================================
# CHATBOT
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Nhập câu hỏi...")

if question:

    st.chat_message("user").write(question)

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    context = ""

    for name, df in data.items():

        context += f"\nSheet: {name}\n"
        context += df.head(20).to_string()

    prompt = f"""
Bạn là trợ lý phân tích dữ liệu ngành điện lực.

Dữ liệu Google Sheets:

{context}

Câu hỏi:

{question}

Nếu có dữ liệu thì phân tích.
Nếu không thì trả lời kiến thức chung.
"""

    response = model.generate_content(prompt)

    answer = response.text

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )