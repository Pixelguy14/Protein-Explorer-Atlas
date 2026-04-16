import os
import logging
from flask import Flask
from .api.routes import load_ai_models

def create_app():
    app = Flask(__name__)
    
    # --- Config ---
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure simple logger
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # --- Context Processors ---
    @app.context_processor
    def inject_global_vars():
        return {
            'app_name': 'Protein Explorer Atlas',
            'app_version': '3.0.0 Zero-Shot VEP'
        }

    # --- Register Blueprints ---
    from .frontend.routes import frontend_bp
    from .api.routes import api_bp

    app.register_blueprint(frontend_bp)
    app.register_blueprint(api_bp)

    # Initialize AI models upon app construction
    with app.app_context():
        load_ai_models()

    return app
