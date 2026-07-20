import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routers import auth_routes, ai_routes

# Create database tables (SQLite by default, or Postgres if DATABASE_URL is set)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Apex Financial Advisor API",
    description="Unified backend with FastAPI, PostgreSQL, and Groq LLM",
    version="2.0.0"
)

# SECURITY: Configure CORS securely
origins = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:5001",
    "http://localhost:5002",
    "https://apex-financial-advisor.onrender.com",
    # Add your production frontend domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # MUST be True for HttpOnly Cookies to work
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_routes.router)
app.include_router(ai_routes.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Unified FastAPI backend is running"}

# Run with: uvicorn main:app --reload
