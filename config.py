"""Flask config."""
import os
from dotenv import load_dotenv
from urllib.parse import urlparse



BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    """Flask configuration variables."""
    FLASK_ENV = os.environ.get("FLASK_ENV")
    #FLASK_APP = wsgi:wsgi.py
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER = "templates"
    COMPRESSOR_DEBUG = os.environ.get("COMPRESSOR_DEBUG")
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    DATABASE_URI = os.environ['DATABASE_URL']
    DATABASE_URI= DATABASE_URI[:8]+'ql' + DATABASE_URI[8:]
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TEMPLATES_AUTO_RELOAD = True
    SESSION_PERMANENT = False
