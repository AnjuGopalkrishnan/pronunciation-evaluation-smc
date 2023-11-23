from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float, MetaData, Table
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://speechparrot_user:gredX4buAaqNMb76PtLs59oGhMMwZDky@dpg-clessbd3qkas73b08k1g-a.singapore-postgres.render.com/speechparrot', echo=True)
metadata = MetaData()
connection = engine.connect()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
db = SessionLocal()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


users = Table(
    'users', metadata,
    Column('user_id', Integer, primary_key=True, index=True),
    Column('username', String),
    Column('email', String),
    Column('password', String),
)

words = Table(
    'words', metadata,
    Column('wordid', Integer, primary_key=True, index=True),
    Column('word', String, primary_key=True),
    Column('weekid', Integer),
)

user_progress = Table(
    'user_progress', metadata,
    Column('username', Integer, primary_key=True),
    Column('word', String, primary_key=True),
    Column('done', String),
    Column('favorite', String),
    Column('score', String)
)


class User(Base):
    __table__ = users

class Word(Base):
    __table__ = words

class Userprogress(Base):
    __table__ = user_progress


def GetDBConnection():
    return connection
