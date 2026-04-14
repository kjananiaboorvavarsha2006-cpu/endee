# Smart Exam Analyzer

> Upload any question paper — AI auto-detects the subject, extracts topics, and generates the most important questions to study.

---

## Demo

<p align="center">
  <img src="assets/demo.png" alt="Smart Exam Analyzer UI" width="100%"/>
</p>

---

## What It Does

- Accepts **any file type** — PDF, DOCX, TXT, CSV, XLSX, JSON, PNG, JPG
- **Auto-detects the subject** from the document content using LLM (no manual selection needed)
- Extracts all **key topics** covered in the paper
- Generates **important exam questions with model answers** for each topic via RAG pipeline
- Provides a **study priority guide** summarizing what to focus on
- Includes an **Ask AI** box to query anything about the subject

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| LLM | Groq — `llama-3.1-8b-instant` |
| PDF Parsing | PyPDF2 |
| Word Parsing | python-docx |
| Excel Parsing | openpyxl |
| Image OCR | pytesseract + Pillow |

---

## Project Structure

```
exam_analyzer/
├── app.py              # Flask server — RAG pipeline, LLM calls, file parsing
├── exam_upload.html    # Frontend UI — drag & drop, results display
├── question_data.py    # Sample past-year question dataset
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Setup

### 1. Get a Groq API Key

Sign up free at [console.groq.com](https://console.groq.com) and copy your API key.

### 2. Add your API key

Open `app.py` and replace the key on line 12:

```python
GROQ_API_KEY = "your_groq_api_key_here"
```

### 3. Install dependencies

```bash
cd exam_analyzer
pip install -r requirements.txt
```

> For image OCR (PNG/JPG files), also install Tesseract on your system:
> - Windows: [tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
> - Mac: `brew install tesseract`
> - Linux: `sudo apt install tesseract-ocr`

### 4. Run the server

```bash
python app.py
```

### 5. Open in browser

```
http://localhost:5000
```

---

## How to Use

1. Open `http://localhost:5000` in your browser
2. Drag and drop your question paper files (or click to browse)
3. You can upload **multiple files at once** — mix PDFs, images, Word docs, etc.
4. Click **Analyze Files**
5. The AI will:
   - Detect the subject automatically
   - List all topics found in the paper
   - Generate important questions + answers for each topic
   - Show a study priority guide
6. Use the **Ask AI** box at the bottom to ask anything about the subject

---

## Supported File Types

| Type | Extensions |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain Text | `.txt`, `.md` |
| Spreadsheet | `.xlsx`, `.xls`, `.csv` |
| JSON | `.json` |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` |

---

## RAG Pipeline

```
Upload Files
     ↓
Extract Text (PyPDF2 / docx / OCR / etc.)
     ↓
Detect Subject  ──→  Groq LLM
     ↓
Extract Topics  ──→  Groq LLM
     ↓
Embed Lines     ──→  SentenceTransformer → ChromaDB
     ↓
Per Topic: Retrieve relevant chunks → Generate Q&A  ──→  Groq LLM
     ↓
Display Results in Browser
```

---

## Requirements

```
flask
chromadb
sentence-transformers
groq
PyPDF2
python-docx
openpyxl
Pillow
pytesseract
```

---

## Notes

- The Groq free tier is sufficient for normal use
- Image OCR requires Tesseract installed separately on your OS
- Works for **any subject** — CS, Physics, Law, Finance, Medicine, etc.
- No data is stored permanently — everything is in-memory per session
