from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flask_cors import CORS

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Load the configuration from Config class
    app.config.from_object(Config)
    
    # Initialize the database with the app
    db.init_app(app)

    # Register the blueprint
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app
