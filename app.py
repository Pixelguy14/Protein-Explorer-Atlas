from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Crear directorio de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Extensiones permitidas para archivos de proteínas
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'pdb', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_protein_sequence(file_path):
    """Lee secuencia de proteína desde archivo FASTA o texto plano"""
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            
        # Si es formato FASTA, extraer solo la secuencia
        if content.startswith('>'):
            lines = content.split('\n')
            sequence = ''.join(line for line in lines[1:] if not line.startswith('>'))
        else:
            # Asumir que es texto plano con la secuencia
            sequence = ''.join(content.split())
            
        return sequence.upper()
    except Exception as e:
        return None

def get_uniprot_from_sequence(sequence):
    """
    Función simulada para obtener UniProt ID desde secuencia
    En implementación real, usarías BLAST o similar
    """
    # Simulación - mapear algunas secuencias conocidas
    known_proteins = {
        'P04637': 'P53_HUMAN',  # p53
        'P01308': 'INS_HUMAN',  # Insulin
        'P68871': 'HBB_HUMAN',  # Hemoglobin
        'P02768': 'ALBU_HUMAN', # Albumin
        'P00750': 'PLAT_HUMAN'  # Tissue plasminogen activator
    }
    
    # Retornar un UniProt ID aleatorio para demo
    return np.random.choice(list(known_proteins.keys()))

def load_domain_data():
    """Cargar datos de dominios Pfam"""
    try:
        # Intentar cargar archivo real, si no existe usar datos simulados
        if os.path.exists('pfam/Pfam-A.regions_human.tsv'):
            domains = pd.read_csv('pfam/Pfam-A.regions_human.tsv', sep='\t',
                                names=['uniprot_id', 'pfam_id', 'start', 'end', 'eval'])
        else:
            # Datos simulados para demo
            domains_data = [
                ['P04637', 'PF00870', 94, 292, 1.2e-50],
                ['P04637', 'PF07710', 319, 393, 2.1e-15],
                ['P01308', 'PF00049', 25, 54, 3.4e-12],
                ['P01308', 'PF00049', 90, 119, 1.8e-10],
                ['P68871', 'PF00042', 7, 146, 5.2e-45],
                ['P02768', 'PF00273', 28, 200, 1.1e-60],
                ['P02768', 'PF00273', 210, 380, 2.3e-58],
                ['P00750', 'PF00051', 87, 176, 4.5e-25]
            ]
            domains = pd.DataFrame(domains_data, 
                                 columns=['uniprot_id', 'pfam_id', 'start', 'end', 'eval'])
        return domains
    except Exception as e:
        print(f"Error loading domain data: {e}")
        return pd.DataFrame()

def load_sifts_data():
    """Cargar mapeo UniProt->PDB"""
    try:
        if os.path.exists('pdb/uniprot_pdb_human.tsv'):
            sifts = pd.read_csv('pdb/uniprot_pdb_human.tsv', sep='\t')
        else:
            # Datos simulados para demo
            sifts_data = [
                ['P04637', '1TUP'],
                ['P04637', '2OCJ'],
                ['P01308', '3I40'],
                ['P01308', '1MSO'],
                ['P68871', '1HHO'],
                ['P02768', '1AO6'],
                ['P00750', '1PML']
            ]
            sifts = pd.DataFrame(sifts_data, columns=['UniProt', 'PDB'])
        return sifts
    except Exception as e:
        print(f"Error loading SIFTS data: {e}")
        return pd.DataFrame()

def predict_protein_family(sequence):
    """
    Función simulada para predicción de familia de proteínas
    En implementación real, aquí cargarías tu modelo entrenado
    """
    # Obtener UniProt ID simulado
    uniprot_id = get_uniprot_from_sequence(sequence)
    
    # Simulación de predicción - reemplazar con tu modelo real
    families = [
        {'family': 'Immunoglobulin', 'confidence': 0.92, 'pdb_id': '1IGY', 'description': 'Antibody heavy chain', 'uniprot_id': uniprot_id},
        {'family': 'Kinase', 'confidence': 0.87, 'pdb_id': '1ATP', 'description': 'Protein kinase domain', 'uniprot_id': uniprot_id},
        {'family': 'Helix-turn-helix', 'confidence': 0.78, 'pdb_id': '1HTH', 'description': 'DNA-binding domain', 'uniprot_id': uniprot_id},
        {'family': 'Beta-barrel', 'confidence': 0.65, 'pdb_id': '1BBL', 'description': 'Membrane protein', 'uniprot_id': uniprot_id},
        {'family': 'Zinc finger', 'confidence': 0.58, 'pdb_id': '1ZNF', 'description': 'DNA-binding protein', 'uniprot_id': uniprot_id}
    ]
    
    # Métricas simuladas
    metrics = {
        'accuracy': 0.94,
        'precision': 0.91,
        'recall': 0.89,
        'f1_score': 0.90,
        'processing_time': np.random.uniform(0.5, 2.0)
    }
    
    return families, metrics, uniprot_id

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'protein_file' not in request.files:
            return jsonify({'error': 'No se encontró archivo'}), 400
        
        file = request.files['protein_file']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Leer secuencia de proteína
            sequence = read_protein_sequence(file_path)
            
            if not sequence:
                return jsonify({'error': 'No se pudo leer la secuencia de proteína'}), 400
            
            # Realizar predicción
            families, metrics, uniprot_id = predict_protein_family(sequence)
            
            # Limpiar archivo temporal
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'sequence_length': len(sequence),
                'families': families,
                'metrics': metrics,
                'uniprot_id': uniprot_id,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error en el procesamiento: {str(e)}'}), 500

@app.route('/plot/<uniprot_id>')
def plot_protein(uniprot_id):
    """Visualizar dominios de proteína"""
    try:
        # Cargar datos de dominios
        domains = load_domain_data()
        
        # Filtrar para la proteína objetivo
        protein_domains = domains[domains['uniprot_id'] == uniprot_id]
        
        if protein_domains.empty:
            return render_template('protein_plot.html',
                                 protein_id=uniprot_id,
                                 domains=[],
                                 error="No se encontraron dominios para esta proteína")
        
        # Convertir a lista de diccionarios y asegurar tipos correctos
        domains_list = []
        for _, row in protein_domains.iterrows():
            domains_list.append({
                'pfam_id': str(row['pfam_id']),
                'start': int(row['start']),
                'end': int(row['end']),
                'eval': float(row['eval'])
            })
        
        return render_template('protein_plot.html',
                             protein_id=uniprot_id,
                             domains=domains_list)
    except Exception as e:
        return render_template('protein_plot.html',
                             protein_id=uniprot_id,
                             domains=[],
                             error=f"Error: {str(e)}")

@app.route('/structure/<uniprot_id>')
def view_structure(uniprot_id):
    """Visualizar estructura 3D"""
    try:
        # Cargar mapeo UniProt->PDB
        sifts = load_sifts_data()
        
        # Buscar PDB ID
        pdb_matches = sifts[sifts['UniProt'] == uniprot_id]
        
        if pdb_matches.empty:
            return render_template('structure_view.html',
                                 protein_id=uniprot_id,
                                 pdb_id=None,
                                 error="No se encontró estructura PDB para esta proteína")
        
        pdb_id = pdb_matches['PDB'].iloc[0]
        
        return render_template('structure_view.html',
                             protein_id=uniprot_id,
                             pdb_id=pdb_id,
                             all_pdbs=pdb_matches['PDB'].tolist())
    except Exception as e:
        return render_template('structure_view.html',
                             protein_id=uniprot_id,
                             pdb_id=None,
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
