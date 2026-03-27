from flask import Flask, render_template, request, jsonify, session
import os
import traceback
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
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Invalid file type. Please upload a PDF'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Delete any old cache for this file to avoid MemoryError
        cache_file = f"{os.path.basename(filepath)}.pkl"
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass

        # Create vector store
        vector_store, multi_level_retriever, raptor_tree = create_vectorstore_from_pdf(filepath)
        
        if vector_store is None:
            return jsonify({'error': 'Failed to process PDF. The file may be empty or a scanned image.'}), 500

        # Store filepath in session
        session['current_pdf'] = filepath
        
        return jsonify({
            'success': True,
            'message': 'PDF processed and AI indexed successfully',
            'filename': filename
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error during upload: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question_route():
    try:
        data = request.get_json()
        query = data.get('question', '')

        if not query:
            return jsonify({'error': 'No question provided'}), 400

        if 'current_pdf' not in session:
            return jsonify({'error': 'Please upload a PDF first'}), 400

        # Load vector store
        filepath = session['current_pdf']
        vector_store, multi_level_retriever, raptor_tree = create_vectorstore_from_pdf(filepath)

        if not vector_store:
            return jsonify({'error': 'Vector store not found. Please re-upload the PDF'}), 400

        # Get answer — always request dict format
        result = answer_question(
            vector_store,
            query,
            multi_level_retriever=multi_level_retriever,
            raptor_tree=raptor_tree,
            return_context=True,
            return_sources=True
        )

        # Handle both dict and str return types
        if isinstance(result, dict):
            return jsonify({
                'success': True,
                'answer': result.get('answer', 'No answer generated.'),
                'context': result.get('context', ''),
                'sources': result.get('sources', [])
            })
        else:
            # result is a plain string (e.g. from cache)
            return jsonify({
                'success': True,
                'answer': str(result),
                'context': '',
                'sources': []
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
