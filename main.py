# main.py
from flask import Flask, render_template, request, jsonify, redirect
import os
import re
import hashlib
import random
from datetime import datetime
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

# --- Config ---
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'pdb', 'txt'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logger simple
logging.basicConfig(level=logging.INFO)
logger = app.logger


# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_protein_id_from_header(header: str | None) -> str | None:
    """
    Intenta extraer un UniProt Accession o ID del encabezado FASTA.
    Soporta:
      >sp|P12345|NAME ...
      >tr|Q9ABC1|NAME ...
      >P12345 texto...
      >ALGUN_ID texto...
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
    Lee un archivo de proteína. Devuelve (sequence_upper, protein_id).
    - Si es FASTA, toma la primera secuencia.
    - Si es texto plano, compacta y lo usa como secuencia.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        header = None
        if content.startswith('>'):
            lines = content.splitlines()
            header = lines[0] if lines else None
            # Toma solo la primera secuencia
            seq_lines = []
            for line in lines[1:]:
                if line.startswith('>'):  # nueva entrada FASTA, rompemos
                    break
                seq_lines.append(line.strip())
            sequence = ''.join(seq_lines)
        else:
            sequence = ''.join(content.split())

        protein_id = extract_protein_id_from_header(header)
        sequence = sequence.upper() if sequence else None
        return sequence, protein_id
    except Exception as e:
        logger.exception("Error al leer archivo de proteína: %s", e)
        return None, None


def predict_protein_family(sequence: str):
    """
    Simulación de predicción (reemplaza por tu modelo real).
    Devuelve una lista de familias con un pdb_id ejemplo y métricas fake.
    """
    families = [
        {'family': 'Immunoglobulin',     'confidence': 0.92, 'pdb_id': '1IGY', 'description': 'Antibody heavy chain'},
        {'family': 'Kinase',             'confidence': 0.87, 'pdb_id': '1ATP', 'description': 'Protein kinase domain'},
        {'family': 'Helix-turn-helix',   'confidence': 0.78, 'pdb_id': '1HTH', 'description': 'DNA-binding domain'},
        {'family': 'Beta-barrel',        'confidence': 0.65, 'pdb_id': '1BBL', 'description': 'Membrane protein'},
        {'family': 'Zinc finger',        'confidence': 0.58, 'pdb_id': '1ZNF', 'description': 'DNA-binding protein'}
    ]
    metrics = {
        'accuracy': 0.94,
        'precision': 0.91,
        'recall': 0.89,
        'f1_score': 0.90,
        'processing_time': round(random.uniform(0.5, 2.0), 3)
    }
    return families, metrics


def get_domains_for_protein(protein_id: str):
    """
    Devuelve dominios de ejemplo para graficar (sustituye por tu integración real con Pfam/InterPro).
    Genera 1-3 dominios con posiciones coherentes.
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
            "eval": float(10 ** (-rnd.randint(2, 8)))  # e-values entre 1e-2 y 1e-8 aprox
        })
    return domains


# --- Routes ---
@app.route('/')
def index():
    # Asegúrate de tener templates/index.html o cambia esta ruta a tu landing real
    return render_template('index.html')


@app.post('/predict')
def predict():
    try:
        if 'protein_file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró archivo'}), 400

        file = request.files['protein_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'}), 400

        if not (file and allowed_file(file.filename)):
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'}), 400

        filename = secure_filename(file.filename)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_')}{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        sequence, protein_id = parse_protein_file(file_path)
        # Limpia el archivo temporal
        try:
            os.remove(file_path)
        except Exception:
            pass

        if not sequence:
            return jsonify({'success': False, 'error': 'No se pudo leer la secuencia de proteína'}), 400

        # Fallback si no hay protein_id en el FASTA
        if not protein_id:
            protein_id = f"anon-{hashlib.md5(sequence.encode()).hexdigest()[:8]}"

        families, metrics = predict_protein_family(sequence)

        return jsonify({
            'success': True,
            'sequence_length': len(sequence),
            'protein_id': protein_id,     # snake_case
            'proteinId': protein_id,      # camelCase (por si tu front lo espera así)
            'families': families,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.exception("Error en /predict: %s", e)
        return jsonify({'success': False, 'error': f'Error en el procesamiento: {str(e)}'}), 500


@app.get('/plot/<protein_id>')
def plot_view(protein_id: str):
    if not protein_id or protein_id.lower() == 'undefined':
        logger.warning("plot_view: protein_id inválido: %r", protein_id)
        return render_template(
            'protein_plot.html',
            protein_id=protein_id,
            domains=[],
            error="Falta el UniProt ID (protein_id) para graficar."
        ), 400

    domains = get_domains_for_protein(protein_id)

    # Renderiza aunque no haya dominios; tu template muestra el mensaje bonito
    return render_template(
        'protein_plot.html',
        protein_id=protein_id,
        domains=domains,
        error=None
    )


@app.get('/structure/<pdb_id>')
def structure_view(pdb_id: str):
    if not pdb_id or pdb_id.lower() == 'undefined':
        logger.warning("structure_view: pdb_id inválido: %r", pdb_id)
        return ("Se requiere un PDB ID válido (no 'undefined').", 400)

    # PDB ID clásico: 4 caracteres alfanuméricos (ej. 1CRN)
    if re.fullmatch(r'[A-Za-z0-9]{4}', pdb_id):
        return redirect(f'https://www.rcsb.org/structure/{pdb_id}')

    return ("Se requiere un PDB ID de 4 caracteres (ej. 1CRN). "
            "Si quieres soportar otros tipos de ID, ajusta la ruta.", 400)

@app.get('/view3d/<pdb_id>')
def view3d(pdb_id: str):
    """Renderiza la página de visualización 3D con un PDB ID."""
    if not pdb_id or not re.fullmatch(r'[A-Za-z0-9]{4}', pdb_id):
        return "PDB ID inválido. Se requiere un ID de 4 caracteres.", 400
    
    # Pasamos el PDB ID a la plantilla. El protein_id no es directamente conocido aquí,
    # pero podemos pasarlo si es necesario en el futuro.
    return render_template('structure_view.html', pdb_id=pdb_id, protein_id=None)

# --- Entry point ---
if __name__ == '__main__':
    # Cambia host/port si lo necesitas
    app.run(debug=True, host='127.0.0.1', port=5000)