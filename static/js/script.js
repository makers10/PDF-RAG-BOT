// File upload handling
document.getElementById('fileInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
});

// Drag and drop
const uploadCard = document.querySelector('.upload-card');

uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.style.borderColor = 'var(--primary)';
});

uploadCard.addEventListener('dragleave', () => {
    uploadCard.style.borderColor = 'var(--border)';
});

uploadCard.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadCard.style.borderColor = 'var(--border)';

    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        uploadFile(file);
    } else {
        showStatus('Please upload a PDF file', 'error');
    }
});

// Upload file function
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    showStatus('Uploading and processing PDF...', 'loading');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showStatus(data.message, 'success');
            document.getElementById('fileName').textContent = `📄 ${data.filename}`;
            document.getElementById('chatSection').style.display = 'block';

            // Scroll to chat section
            setTimeout(() => {
                document.getElementById('chatSection').scrollIntoView({
                    behavior: 'smooth'
                });
            }, 300);
        } else {
            showStatus(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        showStatus('Error uploading file. Please try again.', 'error');
        console.error('Upload error:', error);
    }
}

// Show status message
function showStatus(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = message;
    statusDiv.className = `upload-status ${type}`;
    statusDiv.style.display = 'block';
}

// Ask question
async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    addMessage(question, 'user');
    input.value = '';

    // Show loading
    const loadingId = addLoadingMessage();

    // Disable send button
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        // Remove loading message
        removeLoadingMessage(loadingId);

        if (data.success) {
            addMessage(data.answer, 'bot', {
                context: data.context,
                sources: data.sources
            });
        } else {
            addMessage(data.error || 'Error getting answer', 'bot');
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage('Error: Could not get answer. Please try again.', 'bot');
        console.error('Ask error:', error);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

// Add message to chat
function addMessage(text, type, extras = {}) {
    const messagesDiv = document.getElementById('chatMessages');

    // Remove welcome message if exists
    const welcomeMsg = messagesDiv.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    messageDiv.appendChild(contentDiv);

    // Don't display sources and context - keep it clean

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Add loading message
function addLoadingMessage() {
    const messagesDiv = document.getElementById('chatMessages');
    const loadingId = 'loading-' + Date.now();

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-bot';
    messageDiv.id = loadingId;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingMsg = document.getElementById(loadingId);
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

// Toggle context visibility
function toggleContext(element) {
    const contextDiv = element.nextElementSibling;
    if (contextDiv.style.display === 'none') {
        contextDiv.style.display = 'block';
        element.textContent = '📄 Hide Retrieved Context';
    } else {
        contextDiv.style.display = 'none';
        element.textContent = '📄 View Retrieved Context';
    }
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Enter key to send
document.getElementById('questionInput').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

// Auto-resize textarea
document.getElementById('questionInput').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});
