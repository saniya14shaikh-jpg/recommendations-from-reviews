"""
Database Layer — SQLite via SQLAlchemy
Stores predictions and analytics
"""

import os, json
from datetime import datetime
from sqlalchemy import (create_engine, Column, Integer, String,
                        Float, DateTime, Text)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = os.getenv("DATABASE_URL", "sqlite:///rfr.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False}
                        if "sqlite" in DB_URL else {})
Base   = declarative_base()
Session = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"
    id           = Column(Integer, primary_key=True)
    review_text  = Column(Text)
    cleaned_text = Column(Text)
    sentiment    = Column(String(10))
    label        = Column(Integer)
    confidence   = Column(Float)
    model_used   = Column(String(30), default="classical")
    created_at   = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

def save_prediction(review_text, cleaned_text, sentiment,
                    label, confidence, model_used="classical"):
    session = Session()
    try:
        rec = Prediction(
            review_text=review_text,
            cleaned_text=cleaned_text,
            sentiment=sentiment,
            label=label,
            confidence=confidence,
            model_used=model_used
        )
        session.add(rec)
        session.commit()
        return rec.id
    finally:
        session.close()

def get_all_predictions(limit=200):
    session = Session()
    try:
        rows = session.query(Prediction).order_by(
            Prediction.created_at.desc()).limit(limit).all()
        return [{"id":r.id,
                 "review_text":r.review_text,
                 "sentiment":r.sentiment,
                 "label":r.label,
                 "confidence":r.confidence,
                 "model_used":r.model_used,
                 "created_at":str(r.created_at)} for r in rows]
    finally:
        session.close()

def get_stats():
    session = Session()
    try:
        total = session.query(Prediction).count()
        pos   = session.query(Prediction).filter_by(label=1).count()
        neg   = session.query(Prediction).filter_by(label=0).count()
        all_r = session.query(Prediction).all()
        avg_conf = (sum(r.confidence for r in all_r)/len(all_r)) if all_r else 0
        return {"total":total, "positive":pos, "negative":neg,
                "avg_confidence":round(avg_conf,4)}
    finally:
        session.close()