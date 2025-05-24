import streamlit as st
import pickle
import PyPDF2
import docx
import re
import numpy as np

# Load trained model and components
clf = pickle.load(open("rf_clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
embedder = pickle.load(open("bert_embedder.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

# Extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except:
        return file.read().decode("latin-1")

# Get resume text from uploaded file
def get_resume_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext == "txt":
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file format. Upload PDF, DOCX, or TXT.")

# Predict role suitability
def predict_suitability(text, selected_role):
    cleaned_text = clean_text(text)
    embedding = embedder.encode([cleaned_text])
    tfidf_vec = tfidf.transform([cleaned_text]).toarray()
    features = np.hstack((embedding, tfidf_vec))

    probabilities = clf.predict_proba(features)[0]
    role_index = le.transform([selected_role])[0]
    confidence = probabilities[role_index] * 100
    status = "✅ Suitable" if confidence >= 50 else "❌ Not Suitable"
    return status, confidence

# Streamlit UI
def main():
    st.set_page_config(page_title="Resume Role Suitability", layout="wide")
    st.title("📄 HireMatch AI-ResumeScreening Tool")
    st.markdown("Upload a resume **or** paste text below and select a job role to evaluate how well it fits.")

    selected_role = st.selectbox("🎯 Select Target Role", sorted(le.classes_))

    # Upload option
    uploaded_file = st.file_uploader("📎 Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

    # Text input option
    st.markdown("### ✏️ Or paste resume text directly:")
    typed_text = st.text_area("Paste your resume here", "", height=250)

    resume_text = ""

    if uploaded_file:
        try:
            resume_text = get_resume_text(uploaded_file)
            st.success("Resume text extracted successfully from file!")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    elif typed_text.strip():
        resume_text = typed_text
        st.success("Resume text taken from input box!")

    # Prediction section
    if resume_text:
        if st.checkbox("📄 Show resume text used"):
            st.text_area("Resume Text", resume_text, height=300)

        st.subheader("🔍 Suitability Result:")
        status, confidence = predict_suitability(resume_text, selected_role)
        st.markdown(f"{status} for **{selected_role}** (Confidence: **{confidence:.2f}%**)")

if __name__ == "__main__":
    main()
