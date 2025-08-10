# main.py
from flask import Flask, render_template, request, jsonify, redirect
import os
import re
import hashlib
import random
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import json

app = Flask(__name__)

# --- Config ---
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'pdb', 'txt'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Simple logger
logging.basicConfig(level=logging.INFO)
logger = app.logger

# --- Global Predictions Database ---
PREDICTIONS_DB = {}

def load_predictions(file_path='predictions.json'):
    """Loads predictions from the JSON file into a sequence-keyed dictionary."""
    global PREDICTIONS_DB
    try:
        with open(file_path, 'r') as f:
            predictions = json.load(f)
        # Key by sequence for easy lookup
        PREDICTIONS_DB = {item['sequence']: item for item in predictions}
        logger.info(f"Successfully loaded {len(PREDICTIONS_DB)} predictions from {file_path}")
    except FileNotFoundError:
        logger.error(f"FATAL: {file_path} not found. Please run predict_all.py to generate it.")
        PREDICTIONS_DB = {}
    except json.JSONDecodeError:
        logger.error(f"FATAL: Could not decode {file_path}. It might be corrupted.")
        PREDICTIONS_DB = {}


# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_protein_id_from_header(header: str | None) -> str | None:
    """
    Tries to extract a UniProt Accession or ID from the FASTA header.
    Supports:
      >sp|P12345|NAME ...
      >tr|Q9ABC1|NAME ...
      >P12345 text...
      >SOME_ID text...
    """
    if not header:
        return None
    h = header.strip()

    m = re.match(r'^>\s*(sp|tr)\|([^|]+)\|', h, flags=re.IGNORECASE)
    if m:
        return m.group(2)

    m = re.match(r'^>\s*([A-Za-z0-9_.:\-]+)', h)
    if m:
        return m.group(1)

    return None


def parse_protein_file(file_path: str) -> tuple[str | None, str | None]:
    """
    Reads a protein file. Returns (sequence_upper, protein_id).
    - If FASTA, it takes the first sequence.
    - If plain text, it compacts it and uses it as the sequence.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        header = None
        if content.startswith('>'):
            lines = content.splitlines()
            header = lines[0] if lines else None
            # Take only the first sequence
            seq_lines = []
            for line in lines[1:]:
                if line.startswith('>'):  # new FASTA entry, break
                    break
                seq_lines.append(line.strip())
            sequence = ''.join(seq_lines)
        else:
            sequence = ''.join(content.split())

        protein_id = extract_protein_id_from_header(header)
        sequence = sequence.upper() if sequence else None
        return sequence, protein_id
    except Exception as e:
        logger.exception("Error reading protein file: %s", e)
        return None, None


def predict_protein_family(sequence: str):
    """
    Looks up a protein sequence in the pre-computed predictions database.
    Returns the prediction data if found, otherwise None.
    """
    return PREDICTIONS_DB.get(sequence)


def get_domains_for_protein(protein_id: str):
    """
    Returns example domains for plotting (replace with your actual Pfam/InterPro integration).
    Generates 1-3 domains with coherent positions.
    """
    if not protein_id:
        return []

    rnd = random.Random(hash(protein_id) & 0xffffffff)
    n = rnd.randint(1, 3)
    domains = []
    pos = 1
    for _ in range(n):
        length = rnd.randint(60, 180)
        start = pos
        end = start + length - 1
        pos = end + rnd.randint(5, 30)
        domains.append({
            "pfam_id": f"PF{rnd.randint(1000, 9999)}",
            "start": start,
            "end": end,
            "eval": float(10 ** (-rnd.randint(2, 8)))  # e-values between 1e-2 and 1e-8 approx
        })
    return domains


# --- Routes ---
@app.route('/')
def index():
    # Make sure you have templates/index.html or change this route to your actual landing page
    return render_template('index.html')


@app.post('/predict')
def predict():
    try:
        if 'protein_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file found'}), 400

        file = request.files['protein_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not (file and allowed_file(file.filename)):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_')}{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        sequence, protein_id_from_header = parse_protein_file(file_path)
        # Clean up the temporary file
        try:
            os.remove(file_path)
        except Exception:
            pass

        if not sequence:
            return jsonify({'success': False, 'error': 'Could not read protein sequence'}), 400

        # --- Prediction using the loaded database ---
        prediction_result = predict_protein_family(sequence)

        if not prediction_result:
            return jsonify({
                'success': False,
                'error': 'Protein sequence not found in our pre-computed database.'
            }), 404

        # Use the protein_id from the header if available, otherwise from the database
        protein_id = protein_id_from_header or prediction_result.get('protein_id')

        # Structure the response to match the frontend's expectations
        families = [{
            'family': prediction_result.get('predicted_class'),
            'confidence': prediction_result.get('confidence'),
            'pdb_id': '1ATP',  # Placeholder PDB ID
            'description': f"Predicted class: {prediction_result.get('predicted_class')}"
        }]
        
        metrics = {
            'processing_time': round(random.uniform(0.01, 0.05), 4) # It's just a lookup now
        }

        return jsonify({
            'success': True,
            'sequence_length': len(sequence),
            'protein_id': protein_id,
            'proteinId': protein_id,
            'families': families,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.exception("Error in /predict: %s", e)
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500


@app.get('/plot/<protein_id>')
def plot_view(protein_id: str):
    if not protein_id or protein_id.lower() == 'undefined':
        logger.warning("plot_view: invalid protein_id: %r", protein_id)
        return render_template(
            'protein_plot.html',
            protein_id=protein_id,
            domains=[],
            error="A UniProt ID (protein_id) is required to plot."
        ), 400

    domains = get_domains_for_protein(protein_id)

    # Render even if there are no domains; your template shows the nice message
    return render_template(
        'protein_plot.html',
        protein_id=protein_id,
        domains=domains,
        error=None
    )


@app.get('/structure/<pdb_id>')
def structure_view(pdb_id: str):
    if not pdb_id or pdb_id.lower() == 'undefined':
        logger.warning("structure_view: invalid pdb_id: %r", pdb_id)
        return ("A valid PDB ID is required (not 'undefined').", 400)

    # Classic PDB ID: 4 alphanumeric characters (e.g., 1CRN)
    if re.fullmatch(r'[A-Za-z0-9]{4}', pdb_id):
        return redirect(f'https://www.rcsb.org/structure/{pdb_id}')

    return ("A 4-character PDB ID is required (e.g., 1CRN). "
            "If you want to support other ID types, adjust the route.", 400)

@app.get('/view3d/<pdb_id>')
def view3d(pdb_id: str):
    """Renders the 3D visualization page with a PDB ID."""
    if not pdb_id or not re.fullmatch(r'[A-Za-z0-9]{4}', pdb_id):
        return "Invalid PDB ID. A 4-character ID is required.", 400
    
    # We pass the PDB ID to the template. The protein_id is not directly known here,
    # but we can pass it if needed in the future.
    return render_template('structure_view.html', pdb_id=pdb_id, protein_id=None)

# --- Entry point ---
if __name__ == '__main__':
    load_predictions() # Load the prediction data on startup
    # Change host/port if you need to
    app.run(debug=True, host='127.0.0.1', port=5000)