import streamlit as st
import os

st.set_page_config(page_title="IS Project", page_icon="🤖", layout="wide")

st.markdown(""" 
    <style>
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: #2d2d2d; 
        padding: 20px;
        border-radius: 15px;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.1);
    }

    /* Title Styling */
    .sidebar .sidebar-content h1 {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Radio Button Styling */
    .stRadio>div>label {
        display: flex;
        align-items: center;
        padding: 12px 20px;
        margin-bottom: 12px;
        background-color: #0C0C0C;  /* Slightly lighter color */
        color: white;
        font-size: 18px;
        font-weight: 500;
        border-radius: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        width: 100%;  /* Ensures equal width for all radio buttons */
    }

    .stRadio>div>label:hover {
        background-color: #6a82fb;
    }

    /* Active (selected) radio button styles */
    .stRadio>div>label[aria-checked="true"] {
        background-color: #3b8d99;  /* Highlight for selected item */
        color: white;
    }

    /* Icon and Text Layout */
    .stRadio>div>label>svg {
        margin-right: 10px;
        width: 24px;
        height: 24px;
    }

    /* Add spacing between the sidebar content */
    .sidebar .sidebar-content > div {
        margin-bottom: 10px;
    }

    /* Ensuring equal width for the entire sidebar */
    .sidebar .sidebar-content {
        width: 250px;  /* Fixed width for sidebar */
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📂 Navigation")

pages = {
    "🏠 Home": "custom_pages/home.py",
    "📌 Machine Learning": "custom_pages/machine_learning.py",
    "🧠 Neural Networks": "custom_pages/neural_networks.py",
    "📖 ML Description": "custom_pages/ml_description.py",
    "📖 NN Description": "custom_pages/nn_description.py",
}

selected_page = st.sidebar.radio("", list(pages.keys()))

if "selected_page_file" not in st.session_state or st.session_state.selected_page != selected_page:
    selected_page_file = pages[selected_page]
    st.session_state.selected_page = selected_page
    if os.path.exists(selected_page_file):
        with open(selected_page_file, "r", encoding="utf-8") as file:
            st.session_state.page_content = file.read()

if "page_content" in st.session_state:
    exec(st.session_state.page_content, globals())
