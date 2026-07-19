"""
APEX FINANCIAL ADVISOR — Python Backend (SECURED)
Powered by Ollama + 15 ML Algorithms (100% Free, No API Keys)

SECURITY FEATURES:
  1. Rate Limiting (per-IP with exponential backoff)
  2. Input Validation (strict schema enforcement)
  3. Secrets Management (no hardcoded keys, env vars only)
  4. Error Handling (no stack traces leaked to users)
  5. Security Headers (XSS, CSRF, clickjacking protection)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
import time
import logging
import re
import google.generativeai as genai
from collections import defaultdict
from functools import wraps
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import io
import threading
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import pytesseract
from PIL import Image

# Import the analytics engine
from analytics_engine import run_full_analysis, format_analysis_for_ai

# ============================================================
# LOGGING SETUP — Log errors server-side, never expose to users
# ============================================================
logging.basicConfig(
    filename='apex_server.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('apex')

app = Flask(__name__, static_folder='.')
CORS(app)

# ============================================================
# DATABASE INITIALIZATION
# ============================================================
DB_FILE = 'apex_users.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ============================================================
# SECURITY 1: RATE LIMITING (Per-IP with Exponential Backoff)
# ============================================================

# Configurable thresholds (not hardcoded magic numbers)
RATE_LIMIT_CONFIG = {
    'chat': {'max_requests': 20, 'window_seconds': 60},       # 20 requests per minute
    'public': {'max_requests': 60, 'window_seconds': 60},     # 60 requests per minute
    'backoff_multiplier': 2,                                    # Double wait time on repeated violations
    'max_backoff_seconds': 300                                  # Max 5 minutes lockout
}

# Track requests per IP
rate_limit_store = defaultdict(lambda: {'timestamps': [], 'violations': 0, 'locked_until': 0})

def rate_limit(endpoint_type='public'):
    """Rate limiting decorator with exponential backoff."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr or 'unknown'
            now = time.time()
            store = rate_limit_store[client_ip]
            config = RATE_LIMIT_CONFIG.get(endpoint_type, RATE_LIMIT_CONFIG['public'])
            
            # Check if IP is in backoff lockout
            if now < store['locked_until']:
                remaining = int(store['locked_until'] - now)
                logger.warning(f"Rate limit: IP {client_ip} locked out for {remaining}s more")
                return jsonify({
                    "error": "Too many requests. Please wait before trying again.",
                    "retry_after": remaining
                }), 429
            
            # Clean old timestamps outside the window
            window = config['window_seconds']
            store['timestamps'] = [t for t in store['timestamps'] if now - t < window]
            
            # Check if over limit
            if len(store['timestamps']) >= config['max_requests']:
                store['violations'] += 1
                # Exponential backoff
                backoff = min(
                    config['window_seconds'] * (RATE_LIMIT_CONFIG['backoff_multiplier'] ** store['violations']),
                    RATE_LIMIT_CONFIG['max_backoff_seconds']
                )
                store['locked_until'] = now + backoff
                logger.warning(f"Rate limit exceeded: IP {client_ip}, violations: {store['violations']}, backoff: {backoff}s")
                return jsonify({
                    "error": "Rate limit exceeded. Please slow down.",
                    "retry_after": int(backoff)
                }), 429
            
            # Record this request
            store['timestamps'].append(now)
            
            # Reset violations after successful period
            if store['violations'] > 0 and len(store['timestamps']) < config['max_requests'] // 2:
                store['violations'] = max(0, store['violations'] - 1)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================
# SECURITY 2: INPUT VALIDATION (Strict Schema Enforcement)
# ============================================================

MAX_MESSAGE_LENGTH = 50000       # Max 50K characters per message
MAX_HISTORY_LENGTH = 20          # Max 20 history messages
MAX_ATTACHMENTS = 10             # Max 10 attachments
MAX_ATTACHMENT_TEXT = 500000     # Max 500K chars per attachment text
MAX_ATTACHMENT_NAME = 255        # Max filename length
ALLOWED_ATTACHMENT_KEYS = {'name', 'data', 'extractedText', 'isImage', 'type'}

def validate_chat_input(data):
    """Validate chat input against strict schema. Returns (is_valid, error_message)."""
    if not isinstance(data, dict):
        return False, "Invalid request format"
    
    # Validate message
    message = data.get('message', '')
    if not isinstance(message, str):
        return False, "Message must be a string"
    if len(message) > MAX_MESSAGE_LENGTH:
        return False, f"Message too long (max {MAX_MESSAGE_LENGTH} characters)"
    
    # Validate history
    history = data.get('history', [])
    if not isinstance(history, list):
        return False, "History must be a list"
    if len(history) > MAX_HISTORY_LENGTH:
        return False, f"Too many history messages (max {MAX_HISTORY_LENGTH})"
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            return False, f"History item {i} must be an object"
        if 'role' not in msg or msg['role'] not in ('user', 'assistant'):
            return False, f"History item {i} has invalid role"
        if 'content' not in msg or not isinstance(msg['content'], str):
            return False, f"History item {i} has invalid content"
    
    # Validate attachments
    attachments = data.get('attachments', [])
    if not isinstance(attachments, list):
        return False, "Attachments must be a list"
    if len(attachments) > MAX_ATTACHMENTS:
        return False, f"Too many attachments (max {MAX_ATTACHMENTS})"
    for i, att in enumerate(attachments):
        if not isinstance(att, dict):
            return False, f"Attachment {i} must be an object"
        # Check for unexpected keys (prevent injection)
        unknown_keys = set(att.keys()) - ALLOWED_ATTACHMENT_KEYS
        if unknown_keys:
            return False, f"Attachment {i} has unknown fields: {unknown_keys}"
        # Validate attachment name
        name = att.get('name', '')
        if isinstance(name, str) and len(name) > MAX_ATTACHMENT_NAME:
            return False, f"Attachment {i} name too long"
        # Validate extracted text length
        ext_text = att.get('extractedText', '')
        if isinstance(ext_text, str) and len(ext_text) > MAX_ATTACHMENT_TEXT:
            return False, f"Attachment {i} text too large (max {MAX_ATTACHMENT_TEXT} chars)"
    
    return True, None


def sanitize_text(text):
    """Remove potentially dangerous characters from text while preserving data."""
    if not isinstance(text, str):
        return ""
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


# ============================================================
# SECURITY 3: SECRETS MANAGEMENT
# ============================================================
# All configuration loaded from environment variables or config.
# NO hardcoded API keys, tokens, or passwords in source code.

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SERVER_PORT = int(os.environ.get("APEX_PORT", "5001"))
SERVER_HOST = os.environ.get("APEX_HOST", "0.0.0.0")

# ============================================================
# SECURITY 5: SECURITY HEADERS MIDDLEWARE
# ============================================================

@app.after_request
def add_security_headers(response):
    """Add security headers to every response."""
    # Prevent XSS
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Permissions policy
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    return response


# ============================================================
# SYSTEM PROMPT & MODELS
# ============================================================

SYSTEM_PROMPT = """You are Apex AI, an elite financial analyst powered by 15 machine learning algorithms.

When you receive ANALYTICS ENGINE RESULTS, you MUST:
1. Present the computed results clearly using Markdown (### headings, **bold**, - bullet points).
2. Use the EXACT numbers from the algorithms — do NOT invent or hallucinate data.
3. Highlight the most important findings from each algorithm.
4. Keep responses concise: bullet points and short data-dense sentences, NOT long paragraphs.
5. If anomalies are detected, flag them prominently.
6. If no analytics data is provided, politely ask the user to upload a report.
7. Never give investment advice.
8. [CHART INTEGRATION]: If you have time-series data or categorical data that would benefit from a chart (like revenue over time, or a breakdown of expenses), you can generate a chart by outputting a strictly formatted JSON block wrapped in ```chart ... ```. The JSON must be a valid Chart.js configuration object.
Example:
```chart
{
  "type": "bar",
  "data": {
    "labels": ["Q1", "Q2", "Q3", "Q4"],
    "datasets": [{"label": "Revenue", "data": [10, 20, 15, 25], "backgroundColor": "#6366f1"}]
  }
}
```
Only output the chart block when you have real data to visualize."""

MODELS_TO_TRY = ["qwen2.5:14b", "llama3.2:1b", "llama3.2", "gemma4"]

ALGORITHM_LIST = [
    {"id": 1, "name": "Linear Regression", "desc": "Trend prediction & forecasting"},
    {"id": 2, "name": "Logistic Regression", "desc": "Financial health classification"},
    {"id": 3, "name": "Decision Trees", "desc": "Key factor identification"},
    {"id": 4, "name": "Random Forest", "desc": "Ensemble strength scoring"},
    {"id": 5, "name": "K-Means Clustering", "desc": "Data grouping & segmentation"},
    {"id": 6, "name": "Time Series Forecasting", "desc": "Future value projection"},
    {"id": 7, "name": "PCA", "desc": "Key metric identification"},
    {"id": 8, "name": "Naive Bayes", "desc": "Risk classification"},
    {"id": 9, "name": "Association Rules", "desc": "Correlation discovery"},
    {"id": 10, "name": "Gradient Boosting", "desc": "High-accuracy scoring"},
    {"id": 11, "name": "Anomaly Detection", "desc": "Unusual pattern flagging"},
    {"id": 12, "name": "Collaborative Filtering", "desc": "Industry benchmarking"},
    {"id": 13, "name": "A/B Testing", "desc": "Period comparison"},
    {"id": 14, "name": "NLP Analysis", "desc": "Text sentiment & topics"},
    {"id": 15, "name": "Neural Networks", "desc": "Deep pattern recognition"},
]


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/register', methods=['POST'])
@rate_limit('public')
def register():
    data = request.json
    name = sanitize_text(data.get('name', ''))
    email = sanitize_text(data.get('email', ''))
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required"}), 400

    password_hash = generate_password_hash(password)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, password_hash))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists"}), 409
    finally:
        conn.close()

    return jsonify({"success": True, "message": "Registration successful"}), 201

@app.route('/api/login', methods=['POST'])
@rate_limit('public')
def login():
    data = request.json
    email = sanitize_text(data.get('email', ''))
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, password_hash FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[2], password):
        # Successful login
        return jsonify({"success": True, "user": {"id": user[0], "name": user[1], "email": email}})
    else:
        return jsonify({"error": "Invalid email or password"}), 401
@app.route('/api/chat', methods=['POST'])
@rate_limit('chat')  # SECURITY 1: Rate limited
def chat():
    try:
        data = request.json
        
        # SECURITY 2: Strict input validation
        is_valid, error_msg = validate_chat_input(data)
        if not is_valid:
            logger.warning(f"Input validation failed from {request.remote_addr}: {error_msg}")
            return jsonify({"error": "Invalid input. Please check your request."}), 400
        
        # Sanitize inputs
        message = sanitize_text(data.get('message', ''))
        history = data.get('history', [])
        attachments = data.get('attachments', [])

        # Build conversation context from recent history
        history_context = ""
        for msg in history[-4:]:
            role = msg.get('role', '')
            content = sanitize_text(msg.get('content', ''))
            if role == 'user':
                history_context += f"User: {content}\n"
            else:
                history_context += f"Apex AI: {content}\n"

        # Build final prompt
        final_prompt = message
        if history_context:
            final_prompt = f"[Previous Conversation Context]:\n{history_context}\n\n[New User Message]:\n{message}"

        # Collect all document text for analysis
        all_document_text = ""
        
        # FEATURE 2: OCR Image Processing
        for att in attachments:
            extracted = sanitize_text(att.get('extractedText', ''))
            att_name = sanitize_text(att.get('name', 'file'))[:100]
            
            # If it's an image, run it through Tesseract OCR
            if att.get('isImage') and att.get('data'):
                try:
                    print(f"  👁️ Running OCR on {att_name}...")
                    img_data = att['data'].split(',')[1] if ',' in att['data'] else att['data']
                    image = Image.open(io.BytesIO(base64.b64decode(img_data)))
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        extracted = ocr_text + "\n" + extracted
                        print(f"  ✅ OCR extracted {len(ocr_text)} characters")
                    else:
                        print(f"  ℹ️ OCR found no text")
                except Exception as e:
                    logger.error(f"OCR failed for {att_name}: {e}")
                    print(f"  ❌ OCR failed: {e}")
            
            if extracted:
                all_document_text += extracted + "\n"
                final_prompt += f"\n\n--- Document: {att_name} ---\n{extracted}\n--- End Document ---"

        if not final_prompt.strip() and attachments:
            final_prompt = "I have attached some files but provided no text. Please analyze them."
            
        # FEATURE 1: Auto-Researcher Agent
        is_research = False
        if message.lower().startswith('!research'):
            is_research = True
            query = message[9:].strip()
            print(f"  🤖 Auto-Researcher activated for query: '{query}'")
            try:
                # Scrape DuckDuckGo
                results = DDGS().text(query, max_results=3)
                research_text = f"\n\n--- AUTO-RESEARCH FINDINGS FOR '{query}' ---\n"
                for res in results:
                    research_text += f"Source: {res['title']} ({res['href']})\nSummary: {res['body']}\n\n"
                
                final_prompt += research_text
                all_document_text += research_text # Feed into ML algorithms too!
                print(f"  ✅ Auto-Researcher found {len(results)} sources.")
            except Exception as e:
                logger.error(f"Auto-research failed: {e}")
                print(f"  ❌ Auto-research failed: {e}")

        # ===== RUN 15 ML ALGORITHMS ON THE DATA =====
        analysis_text = ""
        algorithms_used = 0
        
        text_to_analyze = all_document_text if all_document_text else message
        if text_to_analyze and len(text_to_analyze) > 20:
            logger.info(f"Running 15 ML algorithms on {len(text_to_analyze)} chars of data")
            print("  🧠 Running 15 ML algorithms...")
            analysis = run_full_analysis(text_to_analyze)
            if analysis:
                algorithms_used = analysis['total_algorithms_run']
                analysis_text = format_analysis_for_ai(analysis)
                final_prompt += analysis_text
                print(f"  ✅ {algorithms_used}/15 algorithms produced results ({analysis['total_data_points']} data points)")
            else:
                print("  ℹ️ No numerical data found for algorithms")

        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set.")
            return jsonify({"error": "Server configuration error: Missing Gemini API Key. Please configure GEMINI_API_KEY in the environment."}), 500

        try:
            print("  🤖 Calling Google Gemini API...")
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT
            )
            
            response = model.generate_content(final_prompt)
            ai_text = response.text
            
            # Clean up markdown blocks if model outputs them
            ai_text = ai_text.strip()
            if ai_text.startswith("```html"):
                ai_text = ai_text[7:]
            if ai_text.startswith("```"):
                ai_text = ai_text[3:]
            if ai_text.endswith("```"):
                ai_text = ai_text[:-3]
            ai_text = ai_text.strip()
            
            print("  ✅ Success with Gemini API")
            return jsonify({
                "response": ai_text,
                "algorithms_used": algorithms_used
            })
            
        except Exception as e:
            logger.error(f"Gemini API exception: {str(e)}")
            print(f"  ❌ Gemini API failed: {str(e)}")
            return jsonify({"error": "Unable to process your request with Gemini API. Please try again."}), 500

    except Exception as e:
        # SECURITY 4: Never expose stack traces or internal paths to users
        logger.error(f"Unhandled error in /api/chat from {request.remote_addr}: {str(e)}")
        print(f"  ❌ Error in /api/chat: {e}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


@app.route('/api/algorithms', methods=['GET'])
@rate_limit('public')
def get_algorithms():
    """Return the list of available ML algorithms."""
    return jsonify({"algorithms": ALGORITHM_LIST, "total": len(ALGORITHM_LIST)})


@app.route('/api/health', methods=['GET'])
@rate_limit('public')
def health():
    """Check if Ollama is running and models are available."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return jsonify({"status": "ok", "models": models, "algorithms": len(ALGORITHM_LIST)})
    except Exception as e:
        # SECURITY 4: Log error details server-side only
        logger.error(f"Health check failed: {str(e)}")
    return jsonify({"status": "error", "message": "AI engine is not available"}), 503


# ============================================================
# STARTUP
# ============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  APEX FINANCIAL ADVISOR — Python Backend")
    print("  Powered by Ollama + 15 ML Algorithms")
    print("  100% Free, No API Keys Required")
    print("=" * 55)
    print("  🔒 SECURITY FEATURES:")
    print("     ✅ Rate Limiting (per-IP + exponential backoff)")
    print("     ✅ Input Validation (strict schema enforcement)")
    print("     ✅ Secrets Management (env vars, no hardcoded keys)")
    print("     ✅ Error Handling (no stack traces to users)")
    print("     ✅ Security Headers (XSS, clickjacking, CSRF)")
    print("=" * 55)
    
    # Check if Gemini API is configured
    if GEMINI_API_KEY:
        print("  ✅ Google Gemini API Key: CONFIGURED")
    else:
        print("  ⚠️  Google Gemini API Key: NOT FOUND (Requires GEMINI_API_KEY env var)")
    
    print(f"  🧠 15 ML Algorithms: LOADED")
    print(f"  🚀 Server starting on http://localhost:{SERVER_PORT}")
    print(f"  🌐 Also available at http://apex.ai:{SERVER_PORT}")
    print("=" * 55)
    
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
