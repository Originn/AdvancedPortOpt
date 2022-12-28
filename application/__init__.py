from flask import Flask
from flask_assets import Environment
import os
import pylibmc
from flask_session import Session
import redis

def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')
    assets = Environment()
    assets.init_app(app)

    cache_servers = os.environ.get('MEMCACHIER_SERVERS')
    cache_user = os.environ.get('MEMCACHIER_USERNAME')
    cache_pass = os.environ.get('MEMCACHIER_PASSWORD')
    app.config.update(
        SESSION_TYPE = 'redis',
        SESSION_REDIS = redis.from_url(os.environ.get('REDIS_URL')))
    Session(app)

    with app.app_context():
        # Import parts of our core Flask app
        from application import routes
        from .assets import compile_static_assets

        # Import Dash application
        from .dashboard.dashboard import init_dashboard
        #with app.test_request_context():
        app=init_dashboard(app)
        # Compile static assets
        compile_static_assets(assets)

        return app
