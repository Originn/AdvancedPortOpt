from flask import Flask
from flask_assets import Environment
import os
import pylibmc
from flask_session import Session




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
        SESSION_TYPE = 'memcached',
        SESSION_MEMCACHED = pylibmc.Client(cache_servers.split(','), binary=True,
                           username=cache_user, password=cache_pass, behaviors={
                                    # Faster IO
                                    'tcp_nodelay': True,
                                    # Keep connection alive
                                    'tcp_keepalive': True,
                                    # Timeout for set/get requests
                                    'connect_timeout': 2000, # ms
                                    'send_timeout': 750 * 1000, # us
                                    'receive_timeout': 750 * 1000, # us
                                    '_poll_timeout': 2000, # ms
                                    # Better failover
                                    'ketama': True,
                                    'remove_failed': 1,
                                    'retry_timeout': 2,
                                    'dead_timeout': 30,
                               }))
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
