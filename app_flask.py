from flask import Flask, render_template, request, jsonify, session
import os
import pickle
from werkzeug.utils import secure_filename
from rag_pipeline import create_vectorstore_from_pdf, answer_question
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create or load vector store
        vector_store_file = f"{filepath}.pkl"

        if os.path.exists(vector_store_file):
            with open(vector_store_file, "rb") as f:
                vector_store = pickle.load(f)
            message = "Vector store loaded from cache"
        else:
            vector_store = create_vectorstore_from_pdf(filepath, chunk_size=800)
            with open(vector_store_file, "wb") as f:
                pickle.dump(vector_store, f)
            message = "Vector store created successfully"

        # Store filename in session
        session['current_pdf'] = filepath

        return jsonify({
            'success': True,
            'message': message,
            'filename': filename
        })

    return jsonify({'error': 'Invalid file type. Please upload a PDF'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('question', '')

    if not query:
        return jsonify({'error': 'No question provided'}), 400

    if 'current_pdf' not in session:
        return jsonify({'error': 'Please upload a PDF first'}), 400

    # Load vector store
    vector_store_file = f"{session['current_pdf']}.pkl"

    if not os.path.exists(vector_store_file):
        return jsonify({'error': 'Vector store not found. Please re-upload the PDF'}), 400

    with open(vector_store_file, "rb") as f:
        vector_store = pickle.load(f)

    # Get answer
    result = answer_question(
        vector_store,
        query,
        top_k=5,
        similarity_threshold=1.5,
        use_hybrid=True,
        return_sources=True,
        return_context=True
    )

    return jsonify({
        'success': True,
        'answer': result['answer'],
        'context': result['context'],
        'sources': result.get('sources', [])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
