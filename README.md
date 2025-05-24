
# 🧠 AI Resume Screening Tool

An intelligent resume screening system that automatically analyzes resumes and determines whether a candidate is **suitable for a given job role or category** using Natural Language Processing (NLP) and Machine Learning.

---

## 🚀 What It Does

- 📄 Upload resumes in bulk (PDF or CSV)
- 🔍 Analyze candidate skills, experience, and keywords
- 🧠 Use ML model to classify if a candidate is suitable for a role
- 📊 Output results in a user-friendly interface with prediction labels
- ⚡ Speeds up initial screening in the hiring process

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn** – Machine learning
- **BERT** – Resume embeddings
- **Pandas** & **NumPy** – Data processing
- **Streamlit** – Interactive frontend
- **Pickle** – Model saving/loading

---

## 📂 Project Structure

```

AI-Resume-Screening-Tool/
├── data/
│                # Resume PDFs for training/testing
├── Resume/
│   └── Resume.csv           # Resume dataset with labels
├── newtrainmodel.py        # Script to train the ML model
├── newstreamlit.py         # Streamlit frontend to upload and test resumes
├── encoder.pkl             # Label encoder for role categories
├── rf\_clf.pkl              # Trained Random Forest model
├── tfidf.pkl               # TF-IDF vectorizer
├── bert\_embedder.pkl       # Resume embeddings
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

````

---

## 🧪 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vyshnavikatkuri/AI-Resume-Screening-Tool.git
cd AI-Resume-Screening-Tool
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run newstreamlit.py
```

---

## 📈 Model Overview

* **Task**: Binary Classification (Suitable / Not Suitable)
* **Algorithms**: Random Forest + BERT
* **Input**: Resume text
* **Output**: Suitability prediction

### Files used:

* `rf_clf.pkl`: Trained classifier
* `encoder.pkl`: Label encoder
* `tfidf.pkl`: TF-IDF vectorizer
* `bert_embedder.pkl`: BERT embeddings

---

## 🙋‍♀️ Developed by

**Vyshnavi Katkuri**
B.Tech IT Student | ML Enthusiast | Deep Learning Explorer
🔗 [GitHub](https://github.com/vyshnavikatkuri)

---

## ⚠️ Notes

* Files like `Resume.csv` and `bert_embedder.pkl` exceed GitHub's 50MB limit. Use [Git LFS](https://git-lfs.github.com) to handle large files.
* Ensure resume PDFs are cleanly formatted for accurate parsing.

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

```




