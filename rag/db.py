# rag/db.py

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer,
    Float, DateTime, Text, ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String(16), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20))
    document_type = Column(String(50))
    tenant_id = Column(String(50), default="vanaciprime")
    chunk_count = Column(Integer, default=0)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="pending")
    error_message = Column(Text)
    sample_questions = Column(Text)
    loader_used = Column(String(30))

    eval_results = relationship(
        "DocumentEvalResult",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentEvalResult(Base):
    __tablename__ = "document_eval_results"

    id = Column(String(36), primary_key=True)
    document_id = Column(
        String(16),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    eval_date = Column(DateTime, default=datetime.utcnow)
    questions_tested = Column(Integer)
    questions_passed = Column(Integer)
    pass_rate = Column(Float)
    threshold = Column(Float, default=0.80)
    status = Column(String(20))

    document = relationship("Document", back_populates="eval_results")


_engine = None
_session_factory = None


def _ensure_secrets_loaded():
    """Try to load secrets from AWS Secrets Manager if DATABASE_URL is missing."""
    if os.environ.get("DATABASE_URL"):
        return
    try:
        from scripts.aws_secrets import load_secrets_to_env
        load_secrets_to_env()
    except Exception:
        pass


def _get_engine():
    global _engine
    if _engine is None:
        _ensure_secrets_loaded()
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL not set. "
                "Add it to AWS Secrets Manager or .env file."
            )
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def SessionLocal():
    """
    Get a new database session.
    Engine is created lazily on first call — after secrets are loaded.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _session_factory()


def get_session():
    """Get a database session as a context manager."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()