import os.path
from datetime import datetime
import re
import pronouncing

from fastapi import FastAPI, File, Form, HTTPException, Depends, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy_utils import database_exists, create_database, drop_database

from wav2vec2_inference import get_wav2vec2_asr_sb_object
import infra.db
import lib.authenticate
from infra import models
from infra.db import User
from lib import jwt
from pydub import AudioSegment

app = FastAPI()

# Singletons
print(f"Setting app wave2vec2 asr speech brain object globally")
app.state.wave2vec2_asr_brain = get_wav2vec2_asr_sb_object('./ml/config/wave2vec2/hparams/inference.yaml')
# app.state.hubert_asr_brain = singleton_instance
print(f"Created wave2vec2 model singleton!")

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

# fetch favorites for user from database
@app.get("/favorites/{username}")
def get_favorites(username: str, db: Session = Depends(infra.db.get_db)):
    favorites = db.query(infra.db.Userprogress).filter(infra.db.Userprogress.username == username).filter(infra.db.Userprogress.favorite == "true").all()
    return favorites

# get phonemes for any sentence
@app.get("/phonemes/{text}")
def get_phonemes(text: str):
    words = text.split()
    WordToPhn=[]
    for word in words:
        pronunciation_list = pronouncing.phones_for_word(word)[0] # choose the first version of the phoneme
        WordToPhn.append(pronunciation_list)

    SentencePhn=' '.join(WordToPhn)
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
        progress = infra.db.Userprogress(username=user_progress.username, word=user_progress.word, done="false", favorite="false", score="0")

    if action == "done":
        progress.done = user_progress.done
    elif action == "favorite":
        progress.favorite = user_progress.favorite
    elif action == "score":
        progress.score = user_progress.score
        
    try:
        db.add(progress)
        db.commit()
        return {
            "success": True
        }
    except SQLAlchemyError as e:
        print(e)
        return {
            "success": False
        }

@app.post("/predict/pronunciation")
def predict_pronunciation(audio: UploadFile = File(...), text: str = Form(None)):
    print("Text: " + text)
    print("Audio file: " + audio.filename)
    # save the audio file to downloads folder after removing the existing file
    try:
        os.remove("./assets/"+audio.filename)
    except:
        pass
    with open("./assets/"+audio.filename, "xb") as buffer:
        buffer.write(audio.file.read())

    audio_path = "./assets/"+audio.filename
    print("Audio path: " + audio_path)

    # convert webm to wav
    # wav = AudioSegment.from_file(audio_path)
    # getaudio = wav.export("./assets/speech.wav", format="wav")
    # audio_path = "./assets/speech.wav"

    # fetch ground truth for the speech
    canonical_phonemes = get_phonemes(text)

    # evaluate pronunciation score from ML model
    predicted_phonemes, score, stats = app.state.wave2vec2_asr_brain.evaluate_test_audio(audio_path, canonical_phonemes)
    
    return {
        "predicted_phonemes": predicted_phonemes,
        "score": score,
        "stats": stats
    }

# This is test code to see if wav2vec2 actually works
@app.get("/test/wav2vec2")
def test_wave2vec():
    test_audio_path = "./assets/arctic_a0100.wav"
    canonical_phonemes = "sil y uw m ah s t s l iy p sil hh iy er jh d sil"  # actual sentence is 'You must sleep he urged'
    predicted_phonemes, score, stats = app.state.wave2vec2_asr_brain.evaluate_test_audio(test_audio_path, canonical_phonemes)
    return {
        "predicted_phonemes": predicted_phonemes,
        "score": score,
        "stats": stats
    }