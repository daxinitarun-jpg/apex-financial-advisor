from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from database import get_db
import models
import auth

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# Pydantic Schemas
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr

    class Config:
        from_attributes = True

@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(name=user.name, email=user.email, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/login")
def login(response: Response, user: UserLogin, db: Session = Depends(get_db)):
    # Authenticate user
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not auth.verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Generate JWT Token
    access_token = auth.create_access_token(data={"sub": str(db_user.id)})
    
    # SECURITY: Set token in HttpOnly Cookie
    # SameSite=Lax prevents CSRF in most modern browsers. Use 'Strict' for maximum security.
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=60*60*24*7, # 7 days
        expires=60*60*24*7,
        samesite="lax",
        secure=False, # Set to True in production (HTTPS)
    )
    
    return {"message": "Successfully logged in"}

@router.post("/logout")
def logout(response: Response):
    # Clear the cookie
    response.delete_cookie(key="access_token", httponly=True, samesite="lax")
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: models.User = Depends(auth.get_current_user)):
    # Automatically injected via auth.get_current_user which validates the HttpOnly cookie
    return current_user
