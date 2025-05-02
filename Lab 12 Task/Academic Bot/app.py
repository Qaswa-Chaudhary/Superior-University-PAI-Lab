from flask import Flask, render_template, request
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
model_name = os.getenv('MODEL_NAME')
api_key    = os.getenv('API_KEY')
base_url   = os.getenv('BASE_URL')

# Initialize OpenAI client
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

app = Flask(__name__)

stored_chunks  = None
stored_index   = None
uploaded_file = None

# PDF text extraction
def extract_chunks(path_pdf):
    doc = fitz.open(path_pdf)
    chunks = []
    for page in doc:
        text = page.get_text()
        paragraphs = text.split('\n\n')
        chunks.extend(paragraphs)
    return chunks

# Embedding creation
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_embedding = model.encode(chunks)
    dim = len(vector_embedding[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vector_embedding).astype('float32'))
    return index, chunks

# Chunk search
def search_chunks(query, index, chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
    return [chunks[i] for i in I[0]]

# OpenAI chat
def chat_with_bot(context_chunks, question):
    context = '\n\n'.join(context_chunks)
    prompt = f"""Answer the following question based only on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': 'You are an intelligent Academic Research Assistant Bot'},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    global stored_chunks, stored_index, uploaded_file
    answer = ""
    
    if request.method == 'POST':
        upload_new = 'upload_new' in request.form
        if upload_new:
            stored_chunks = None
            stored_index = None
            uploaded_file = None
            return render_template('index.html', answer="", uploaded_file=None)

        file = request.files.get('pdf')
        question = request.form.get('question')

        if file and file.filename != '':
            uploaded_file = file.filename
            path = os.path.join('uploads', file.filename)
            file.save(path)
            stored_chunks = extract_chunks(path)
            stored_index, stored_chunks = create_embeddings(stored_chunks)

        if stored_chunks and stored_index and question:
            relevant_chunks = search_chunks(question, stored_index, stored_chunks)
            answer = chat_with_bot(relevant_chunks, question)

    return render_template('index.html', answer=answer, uploaded_file=uploaded_file)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
