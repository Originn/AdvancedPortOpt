"""Flask config."""
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from sqlalchemy import create_engine



BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    """Flask configuration variables."""
    FLASK_ENV = os.environ.get("FLASK_ENV")
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER = "templates"
    COMPRESSOR_DEBUG = os.environ.get("COMPRESSOR_DEBUG")
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    TEMPLATES_AUTO_RELOAD = True
    SESSION_PERMANENT = False
    DATABASE_URI = os.environ['BIT_DATABASE_URL']
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    engine = create_engine(DATABASE_URI,
                           pool_size=10,
                           max_overflow=5,
                           pool_timeout=30)
    SQLALCHEMY_ENGINE = engine
