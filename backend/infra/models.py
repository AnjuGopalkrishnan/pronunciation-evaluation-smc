from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field


class User(BaseModel):
    password: str
    username: str
    email: str

class Words(BaseModel):
    word: str


class Userprogress(BaseModel):
    username: str
    word: str
    done: str
    favorite: str
    score: str
