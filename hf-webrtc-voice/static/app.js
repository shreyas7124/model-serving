// WebRTC Voice Chat Client
let socket;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recognition;
let selectedModel = null;

// Initialize Speech Recognition
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        addMessage('user', transcript);
        socket.emit('voice_message', { text: transcript, model: getSelectedModel() });
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        updateStatus('Error: ' + event.error);
        stopRecording();
    };
    
    recognition.onend = () => {
        if (isRecording) {
            recognition.start(); // Restart if still recording
        }
    };
}


function getSelectedModel() {
    const sel = document.getElementById('model-select');
    if (sel && sel.value) return sel.value;
    return selectedModel;
}

async function loadModels() {
    const sel = document.getElementById('model-select');
    if (!sel) return;
    try {
        const response = await fetch('/api/models', { credentials: 'include' });
        const data = await response.json();
        sel.innerHTML = '';
        const items = (data && data.data) || [];
        const def = (data && data.default) || (items[0] && items[0].id) || '';
        items.forEach((m) => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.label || m.id;
            if (m.id === def) opt.selected = true;
            sel.appendChild(opt);
        });
        selectedModel = sel.value || def;
        sel.onchange = () => {
            selectedModel = sel.value;
            if (socket && socket.connected) {
                socket.emit('start_conversation', { model: selectedModel });
            }
        };
    } catch (e) {
        console.error('Failed to load models', e);
    }
}

// Tab switching
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

// Authentication functions
async function login() {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('Login successful!', 'success');
            setTimeout(() => showChatInterface(data.username), 500);
        } else {
            showMessage(data.error, 'error');
        }
    } catch (error) {
        showMessage('Connection error', 'error');
    }
}

async function register() {
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    const confirm = document.getElementById('register-confirm').value;
    
    if (password !== confirm) {
        showMessage('Passwords do not match', 'error');
        return;
    }
    
    if (password.length < 6) {
        showMessage('Password must be at least 6 characters', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('Registration successful! Please login.', 'success');
            showTab('login');
        } else {
            showMessage(data.error, 'error');
        }
    } catch (error) {
        showMessage('Connection error', 'error');
    }
}

async function guestLogin() {
    try {
        const response = await fetch('/api/auth/guest', {
            method: 'POST',
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showChatInterface(data.username);
        } else {
            showMessage('Error starting guest session', 'error');
        }
    } catch (error) {
        showMessage('Connection error', 'error');
    }
}

async function logout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST',
            credentials: 'include'
        });
        
        if (socket) {
            socket.disconnect();
        }
        
        document.getElementById('chat-container').style.display = 'none';
        document.getElementById('auth-container').style.display = 'block';
        document.getElementById('messages').innerHTML = '';
    } catch (error) {
        console.error('Logout error:', error);
    }
}

function showMessage(message, type) {
    const msgDiv = document.getElementById('auth-message');
    msgDiv.textContent = message;
    msgDiv.className = type;
}

async function showChatInterface(username) {
    document.getElementById('auth-container').style.display = 'none';
    document.getElementById('chat-container').style.display = 'flex';
    document.getElementById('username-display').textContent = username;
    await loadModels();
    initializeSocket();
}

// Socket.IO initialization
function initializeSocket() {
    socket = io('http://localhost:5001', {
        withCredentials: true
    });
    
    socket.on('connect', () => {
        console.log('Connected to server');
        socket.emit('start_conversation', { model: getSelectedModel() });
        addMessage('system', 'Connected! Start speaking or typing.');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        addMessage('system', 'Disconnected from server');
    });
    
    socket.on('ai_response', (data) => {
        addMessage('assistant', data.text);
        speakText(data.text);
    });
    
    socket.on('error', (data) => {
        addMessage('system', 'Error: ' + data.message);
    });
}

// Message handling
function addMessage(role, content) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.textContent = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function sendTextMessage() {
    const input = document.getElementById('text-input');
    const text = input.value.trim();
    
    if (text && socket) {
        addMessage('user', text);
        socket.emit('text_message', { text, model: getSelectedModel() });
        input.value = '';
    }
}

// Voice recording
function toggleRecording() {
    if (!recognition) {
        alert('Speech recognition is not supported in your browser');
        return;
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    try {
        recognition.start();
        isRecording = true;
        
        const btn = document.getElementById('record-btn');
        btn.classList.add('recording');
        document.getElementById('record-text').textContent = 'Stop Recording';
        document.getElementById('record-icon').textContent = '⏹️';
        
        updateStatus('Listening...');
    } catch (error) {
        console.error('Error starting recording:', error);
        updateStatus('Error starting recording');
    }
}

function stopRecording() {
    if (recognition) {
        recognition.stop();
    }
    isRecording = false;
    
    const btn = document.getElementById('record-btn');
    btn.classList.remove('recording');
    document.getElementById('record-text').textContent = 'Start Recording';
    document.getElementById('record-icon').textContent = '🎤';
    
    updateStatus('');
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

// Text-to-Speech
function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        speechSynthesis.speak(utterance);
    }
}

// Handle Enter key in text input
document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    if (textInput) {
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendTextMessage();
            }
        });
    }
});
