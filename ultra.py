

import streamlit as st
from transformers import pipeline
import re
from io import StringIO
from PyPDF2 import PdfReader

# Page config
st.set_page_config(page_title="AI Indian Legal Document Summarizer", page_icon="📜", layout="centered")
st.title("📜 AI Indian Legal Document Summarizer")
st.markdown("##### ✨ Summarize complex legal documents into simple, easy-to-understand bullet points.")

# Load model
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    # Fetch the Hugging Face token from Streamlit secrets
    hf_token = st.secrets["HF_TOKEN"]
    
    # Use the token to load the model
    return pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=hf_token)

summarizer = load_summarizer()


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    input_method = st.radio("Input method:", ("Paste Text", "Upload File"))
    num_bullets = st.slider("Number of bullet points:", 3, 10, 5)
   
# Get text
text = ""
if input_method == "Paste Text":
    text = st.text_area("Paste your legal document here:", height=250)
else:
    uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = "".join(page.extract_text() for page in reader.pages)
        else:
            text = uploaded_file.read().decode("utf-8")

# Summarize on button click
if st.button("🚀 Summarize"):
    if not text:
        st.error("⚠️ Please provide text or upload a file first.")
        st.stop()
    with st.spinner("Generating summary..."):
        # split input into chunks if too long
        max_input = 1024
        chunks = [text[i:i+max_input] for i in range(0, len(text), max_input)]
        results = []
        for chunk in chunks:
            res = summarizer(chunk, max_length=num_bullets*60, min_length=30, do_sample=False)
            results.append(res[0]['summary_text'])
        final_summary = " ".join(results)

        # split summary into sentences
        bullets = re.findall(r'[^\.\!?]+[\.\!?]', final_summary)
        bullets = [b.strip() for b in bullets]
        # if fewer than desired, pad with original sentences
        if len(bullets) < num_bullets:
            orig = re.findall(r'[^\.\!?]+[\.\!?]', text)
            orig = [s.strip() for s in orig]
            for s in orig:
                if s not in bullets:
                    bullets.append(s)
                if len(bullets) >= num_bullets:
                    break
        bullets = bullets[:num_bullets]
        summary_text = "\n".join(f"- {b}" for b in bullets)

    # display summary
    st.subheader("📋 Simplified Summary:")
    for b in bullets:
        st.markdown(f"- {b}")

    # download button
    st.download_button(
        label="📥 Download Summary as .txt",
        data=summary_text,
        file_name="legal_summary.txt",
        mime="text/plain"
    )


