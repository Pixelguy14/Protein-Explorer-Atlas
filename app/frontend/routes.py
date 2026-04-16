from flask import Blueprint, render_template, request, current_app
import re

frontend_bp = Blueprint('frontend', __name__)

@frontend_bp.route('/')
def index():
    return render_template('frontend/index.html')

@frontend_bp.route('/view3d/<pdb_id>')
def view3d(pdb_id: str):
    """
    Renders the 3D molecular viewer.
    Accepts optional query params:
      - url: direct PDB file URL (for AlphaFold sources)
      - source: 'RCSB' or 'AlphaFold' (for UI badge)
    """
    if not pdb_id or not re.fullmatch(r'[A-Za-z0-9_\-]+', pdb_id):
        return "Invalid structure ID.", 400

    pdb_url = request.args.get('url', None)
    source = request.args.get('source', 'RCSB')

    return render_template(
        'frontend/structure_view.html',
        pdb_id=pdb_id,
        pdb_url=pdb_url,
        source=source
    )
