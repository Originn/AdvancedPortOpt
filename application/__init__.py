from flask import Flask
from flask_assets import Environment

def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')
    assets = Environment()
    assets.init_app(app)

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes
        from .assets import compile_static_assets

        # Import Dash application
        from .dashboard.dashboard import init_dashboard
        #with app.test_request_context():

        app=init_dashboard(app)
        # Compile static assets
        compile_static_assets(assets)

        return app
