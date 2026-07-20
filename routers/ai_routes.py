import os
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any
from groq import Groq
import auth

# Import the existing ML and Auto-Researcher logic from the legacy app
from analytics_engine import run_full_analysis, format_analysis_for_ai
from duckduckgo_search import DDGS
import pytesseract
from PIL import Image
import base64
import io
import logging

router = APIRouter(
    prefix="/api",
    tags=["AI Chatbot"]
)

# Groq Setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY is not set!")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class Message(BaseModel):
    role: str
    content: str

class Attachment(BaseModel):
    name: str
    extractedText: Optional[str] = ""
    isImage: Optional[bool] = False
    data: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    attachments: List[Attachment] = []

SYSTEM_PROMPT = """You are Apex AI, an elite financial analyst powered by 15 machine learning algorithms.

When you receive ANALYTICS ENGINE RESULTS, you MUST:
1. Present the computed results clearly using Markdown.
2. Use the EXACT numbers from the algorithms — do NOT invent or hallucinate data.
3. Highlight the most important findings from each algorithm.
4. Keep responses concise: bullet points and short data-dense sentences, NOT long paragraphs.
5. If anomalies are detected, flag them prominently.
6. If no analytics data is provided, politely ask the user to upload a report.
7. Never give investment advice."""

@router.post("/chat")
def chat(payload: ChatRequest, current_user = Depends(auth.get_current_user)):
    """Secure endpoint that requires HttpOnly JWT authentication"""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key missing")

    message = payload.message
    history = payload.history
    attachments = payload.attachments
    
    # 1. Process Attachments and OCR
    all_document_text = ""
    for att in attachments:
        extracted = att.extractedText
        
        # OCR Image Processing
        if att.isImage and att.data:
            try:
                img_data = att.data.split(',')[1] if ',' in att.data else att.data
                image = Image.open(io.BytesIO(base64.b64decode(img_data)))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    extracted = ocr_text + "\n" + (extracted or "")
            except Exception as e:
                logging.error(f"OCR failed for {att.name}: {e}")
        
        if extracted:
            all_document_text += f"\n\n--- Document: {att.name} ---\n{extracted}\n--- End Document ---"
            
    # 2. Auto-Researcher
    research_text = ""
    if message.lower().startswith('!research'):
        query = message[9:].strip()
        try:
            results = DDGS().text(query, max_results=3)
            research_text = f"\n\n--- AUTO-RESEARCH FINDINGS FOR '{query}' ---\n"
            for res in results:
                research_text += f"Source: {res['title']} ({res['href']})\nSummary: {res['body']}\n\n"
        except Exception as e:
            logging.error(f"Auto-research failed: {e}")

    # 3. ML Analytics Engine
    analysis_text = ""
    algorithms_used = 0
    text_to_analyze = all_document_text + research_text
    if len(text_to_analyze) > 20:
        analysis = run_full_analysis(text_to_analyze)
        if analysis:
            algorithms_used = analysis.get('total_algorithms_run', 0)
            analysis_text = format_analysis_for_ai(analysis)

    # 4. Construct Final Prompt
    final_prompt = message
    if all_document_text:
        final_prompt += all_document_text
    if research_text:
        final_prompt += research_text
    if analysis_text:
        final_prompt += analysis_text
        
    # Build Groq Messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history
    for msg in history[-4:]:
        messages.append({"role": msg.role, "content": msg.content})
        
    # Add new message
    messages.append({"role": "user", "content": final_prompt})
    
    try:
        # Call Llama 3 on Groq
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=4096,
        )
        ai_text = chat_completion.choices[0].message.content

        return {
            "response": ai_text,
            "algorithms_used": algorithms_used
        }
    except Exception as e:
        logging.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI processing error")
