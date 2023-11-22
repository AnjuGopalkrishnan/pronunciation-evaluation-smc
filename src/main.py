import os.path
from datetime import datetime
import re
import pronouncing

from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy_utils import database_exists, create_database, drop_database

import infra.db
import lib.authenticate
from infra import models
from infra.db import User
from lib import jwt

app = FastAPI()

# Singletons

# app.state.wave2vec2_asr_brain = singleton_instance
# app.state.hubert_asr_brain = singleton_instance

# Set up CORS
origins = ["*"]  # Update this with your allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Or specify the HTTP methods you allow
    allow_headers=["*"],  # Or specify the HTTP headers you allow
)

# fetch words of the week from database
@app.get("/words")
def get_weekly_words(skip: int = 0, limit: int = None, db: Session = Depends(infra.db.get_db)):
    words =  db.query(infra.db.Word).offset(skip).all()
    if limit is not None:
        words = words[:limit]
    return words

# get phonemes for any sentence
@app.get("/phonemes/{text}")
def get_phonemes(text: str):
    words = text.split()
    WordToPhn=[]
    for word in words:
        pronunciation_list = pronouncing.phones_for_word(word)[0] # choose the first version of the phoneme
        WordToPhn.append(pronunciation_list)

    SentencePhn=' sil '.join(WordToPhn) 
    Output = re.sub(r'\d+', '', SentencePhn) #Remove the digits in phonemes
    return Output.lower()

@app.post("/auth/register")
def register(user: models.User, db: Session = Depends(infra.db.get_db)):
    hashPwd = lib.authenticate.get_password_hash(user.password)

    new_user = User(username=user.username, email=user.email, password=hashPwd)
    db.add(new_user)
    try:
        db.commit()
        return {
            "username": user.username,
            "email": user.email,
            "register_success": True
        }
    except SQLAlchemyError as e:
        print(e)
        return {
            "register_success": False
        }

@app.post("/auth/login")
def login(user: models.LoginUser):
    values = {
        'email': user.email,
    }
    query = text("SELECT * FROM users WHERE email = :email LIMIT 1")
    with infra.db.engine.begin() as conn:
        res = conn.execute(query, values)

    row = res.fetchone()
    if row is None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
        )

    if not lib.authenticate.verify_password(user.password, row[2]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
        )
    #    print("user is ", user)
    access_token = jwt.create_access_token(payload={"email": user.email})
    return {"username": row[1], "access_token": access_token}

@app.get("/user/progress/{username}")
def get_user_progress(username: str, db: Session = Depends(infra.db.get_db)):
    progress = db.query(infra.db.Userprogress).filter(infra.db.Userprogress.username == username).all()
    if not progress:
        raise HTTPException(status_code=404, detail="User profile not found")
    return progress

@app.patch("/user/progress/{action}")
def update_user_progress(action:str, user_progress: models.Userprogress, db: Session = Depends(infra.db.get_db)):
    progress = db.query(infra.db.Userprogress).filter(infra.db.Userprogress.username == user_progress.username).filter(infra.db.Userprogress.word == user_progress.word).first()
    if not progress:
        raise HTTPException(status_code=404, detail="User profile not found")

    if action == "done":
        progress.done = user_progress.done
    elif action == "favorite":
        progress.favorite = user_progress.favorite
    elif action == "score":
        progress.score = user_progress.score
        
    try:
        db.commit()
        return {
            "success": True
        }
    except SQLAlchemyError as e:
        print(e)
        return {
            "success": False
        }

#TODO. get this working first
@app.get("/test/wav2vec")
def test_wave2vec():
    return {
        "info": "yet to implement"
    }
    # test_audio_path = "/content/arctic_a0001.wav"
    # canonical_phonemes = "sil f a m ah s t s l iy p sil hh iy er jh d sil sil"
    #
    # # predicted_phonemes, score, stats = expose_asr_evaluation(test_audio_path, canonical_phonemes)
    #
    # # Print or use the results as needed
    # print(f'Predicted Phonemes: {predicted_phonemes}')
    # print(f'Score: {score}')
    # print(f'Stats: {stats}')