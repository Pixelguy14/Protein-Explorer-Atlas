"""
Protein Explorer Atlas — Zero-Shot Inference Engine
Uses EsmForMaskedLM for Variant Effect Prediction (Masked Marginal LLR)
and Conservation Analysis (per-position Shannon Entropy over 20-AA vocabulary).
No custom training. No downstream CNN. Pure foundation model inference.
"""

from flask import Blueprint, request, jsonify, current_app
import os
import time
import hashlib
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from werkzeug.utils import secure_filename
import re
import requests

api_bp = Blueprint('api', __name__, url_prefix='/api')

ALLOWED_EXTENSIONS = {'fasta', 'fa', 'pdb', 'txt'}
MAX_SEQUENCE_LENGTH = 1022  # ESM-2 context window is 1024 tokens; minus <cls> and <eos>
MAX_BATCHED_MUTATIONS = 20  # Max mutations in a single batched forward pass
ENTROPY_HOTSPOT_THRESHOLD = 1.5  # bits — positions below this are "conserved"
VALID_AA_REGEX = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# ============================================================
# AI CONTEXT — Zero-Shot Foundation Model
# ============================================================
AI_CONTEXT = {
    'tokenizer': None,
    'esm_model': None,   # EsmForMaskedLM (base + LM head)
    'aa_token_ids': None, # Token IDs for the 20 standard amino acids
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def load_ai_models():
    """Loaded once on app boot. No custom weights — pure foundation model."""
    try:
        device = AI_CONTEXT['device']
        current_app.logger.info(f"Loading ESM-2 foundation model on {device}...")
        model_name = "facebook/esm2_t12_35M_UR50D"

        AI_CONTEXT['tokenizer'] = AutoTokenizer.from_pretrained(model_name)
        AI_CONTEXT['esm_model'] = EsmForMaskedLM.from_pretrained(model_name).to(device)
        AI_CONTEXT['esm_model'].eval()

        # Pre-compute the 20 standard amino acid token IDs from the vocabulary
        vocab = AI_CONTEXT['tokenizer'].get_vocab()
        AI_CONTEXT['aa_token_ids'] = [vocab[aa] for aa in STANDARD_AAS if aa in vocab]
        current_app.logger.info(f"Mapped {len(AI_CONTEXT['aa_token_ids'])} standard AAs to token IDs.")

        current_app.logger.info("Zero-shot inference engine ready. No downstream models loaded.")
    except Exception as e:
        current_app.logger.error(f"FATAL: Failed to load ESM-2. Inference will fail. Reason: {e}")


# ============================================================
# CORE ALGORITHMS — Zero-Shot VEP & Conservation
# ============================================================

def compute_conservation_entropy(sequence: str) -> list[float]:
    """
    Per-position Shannon Entropy from unmasked MLM logits over 20 standard AAs.
    Low entropy = conserved (functional domain core).
    High entropy = variable (loop regions, surface residues).
    Single forward pass on the wild-type sequence.
    """
    tokenizer = AI_CONTEXT['tokenizer']
    model = AI_CONTEXT['esm_model']
    device = AI_CONTEXT['device']
    aa_ids = AI_CONTEXT['aa_token_ids']

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH + 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]  # (seq_len+2, vocab_size)

    # Strip [CLS] and [EOS] tokens
    logits = logits[1:-1, :]  # (seq_len, vocab_size)

    # Filter to just the 20 standard AA columns
    aa_logits = logits[:, aa_ids]  # (seq_len, 20)
    probs = torch.softmax(aa_logits, dim=1)  # (seq_len, 20)

    # Shannon entropy in bits: H = -Σ p·log₂(p)
    log_probs = torch.log2(probs + 1e-10)
    entropy = (-torch.sum(probs * log_probs, dim=1)).cpu().numpy().tolist()

    return [round(h, 4) for h in entropy]


def compute_variant_effect(wt_seq: str, mut_seq: str) -> list[float]:
    """
    Masked Marginal Log-Likelihood Ratio per mutated position.
    For each mutation: mask that position in the WT, run MLM head,
    compute LLR = log P(mut_AA) - log P(wt_AA).
    
    Negative LLR = destabilizing (mutant is less likely than wild-type).
    Positive LLR = tolerated/beneficial.
    Zero = unchanged position.
    
    Sparse: only mutated positions have non-zero values.
    """
    tokenizer = AI_CONTEXT['tokenizer']
    model = AI_CONTEXT['esm_model']
    device = AI_CONTEXT['device']
    vocab = tokenizer.get_vocab()
    mask_id = tokenizer.mask_token_id

    # Identify mutated positions
    min_len = min(len(wt_seq), len(mut_seq))
    mutations = []
    for i in range(min_len):
        if wt_seq[i] != mut_seq[i]:
            mutations.append((i, wt_seq[i], mut_seq[i]))

    if not mutations:
        return [0.0] * len(wt_seq)

    llr_array = [0.0] * min_len

    # Tokenize the wild-type ONCE
    base_inputs = tokenizer(wt_seq, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH + 2)
    base_ids = base_inputs['input_ids']       # (1, seq_len+2)
    base_mask = base_inputs['attention_mask']  # (1, seq_len+2)

    # Filter to valid, batchable mutations
    valid_mutations = []
    for pos, wt_aa, mut_aa in mutations[:MAX_BATCHED_MUTATIONS]:
        token_pos = pos + 1  # +1 for [CLS]
        if token_pos >= base_ids.shape[1] - 1:  # Don't mask [EOS]
            continue
        wt_id = vocab.get(wt_aa)
        mut_id = vocab.get(mut_aa)
        if wt_id is None or mut_id is None:
            continue
        valid_mutations.append((pos, token_pos, wt_id, mut_id))

    if not valid_mutations:
        return llr_array

    # Batch: create one copy of input_ids per mutation, mask the target position
    batch_ids = base_ids.repeat(len(valid_mutations), 1).to(device)   # (N, seq_len+2)
    batch_mask = base_mask.repeat(len(valid_mutations), 1).to(device)  # (N, seq_len+2)

    token_positions = []
    for batch_idx, (pos, token_pos, wt_id, mut_id) in enumerate(valid_mutations):
        batch_ids[batch_idx, token_pos] = mask_id
        token_positions.append(token_pos)

    # Single batched forward pass
    with torch.no_grad():
        all_logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits  # (N, seq_len+2, vocab)

    # Extract LLR for each mutation from its masked position
    for batch_idx, (pos, token_pos, wt_id, mut_id) in enumerate(valid_mutations):
        logits_at_pos = all_logits[batch_idx, token_pos]
        probs = torch.softmax(logits_at_pos, dim=0)
        llr = (torch.log(probs[mut_id] + 1e-10) - torch.log(probs[wt_id] + 1e-10)).item()
        llr_array[pos] = round(llr, 4)

    return llr_array


def compute_pseudo_perplexity(sequence: str) -> float:
    """
    Sequence naturalness score. Lower = more natural protein.
    Uses unmasked context (pseudo-likelihood, not true perplexity).
    Returned as a 0-100% "naturalness" scale for UI display.
    """
    tokenizer = AI_CONTEXT['tokenizer']
    model = AI_CONTEXT['esm_model']
    device = AI_CONTEXT['device']

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH + 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, 1:-1, :]  # Strip CLS/EOS

    # Get probability of the actual token at each position
    input_ids = inputs['input_ids'][0, 1:-1]  # Strip CLS/EOS
    probs = torch.softmax(logits, dim=1)
    token_probs = probs[torch.arange(len(input_ids)), input_ids.to(device)]

    # Average negative log-likelihood → perplexity
    nll = -torch.log(token_probs + 1e-10).mean().item()
    ppl = float(np.exp(nll))

    # Map to 0-100% naturalness (PPL=1 → 100%, PPL≥20 → 0%)
    naturalness = max(0.0, min(100.0, (1.0 - (ppl - 1.0) / 19.0) * 100.0))
    return round(naturalness, 1)


# ============================================================
# STRUCTURAL DISCOVERY — RCSB + AlphaFold Fallback
# ============================================================

def get_structure_url(protein_id: str) -> dict | None:
    """
    Two-tier structural lookup:
    1. RCSB PDB (experimental structures)
    2. AlphaFold EBI (predicted structures, ~200M coverage)
    Returns { id, source, pdb_url } or None.
    """
    if not protein_id or protein_id.startswith("GEN_"):
        return None

    # --- Tier 1: RCSB ---
    # Check if the ID itself looks like a 4-char PDB ID
    if re.fullmatch(r'[A-Za-z0-9]{4}', protein_id):
        pdb_url = f"https://files.rcsb.org/download/{protein_id.upper()}.pdb"
        current_app.logger.info(f"Direct PDB ID detected: {protein_id}")
        return {'id': protein_id.upper(), 'source': 'RCSB', 'pdb_url': pdb_url}

    # UniProt-style ID → search RCSB
    query_payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.uniprot_ids",
                "operator": "exact_match",
                "value": protein_id.upper()
            }
        },
        "return_type": "entry"
    }

    try:
        current_app.logger.info(f"RCSB search for UniProt: {protein_id}")
        resp = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query_payload, timeout=5)
        if resp.status_code == 200:
            results = resp.json().get("result_set", [])
            if results:
                pdb_id = results[0]["identifier"]
                current_app.logger.info(f"RCSB hit: {pdb_id}")
                return {
                    'id': pdb_id,
                    'source': 'RCSB',
                    'pdb_url': f"https://files.rcsb.org/download/{pdb_id}.pdb"
                }
        current_app.logger.info(f"RCSB miss for {protein_id}. Trying AlphaFold...")
    except Exception as e:
        current_app.logger.error(f"RCSB query error: {e}")

    # --- Tier 2: AlphaFold ---
    try:
        af_resp = requests.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{protein_id.upper()}",
            timeout=5
        )
        if af_resp.status_code == 200:
            data = af_resp.json()
            if data and len(data) > 0:
                entry = data[0]
                af_pdb_url = entry.get('pdbUrl', '')
                af_id = entry.get('entryId', protein_id)
                current_app.logger.info(f"AlphaFold hit: {af_id}")
                return {'id': af_id, 'source': 'AlphaFold', 'pdb_url': af_pdb_url}
        current_app.logger.warning(f"AlphaFold miss for {protein_id}. No structural data available.")
    except Exception as e:
        current_app.logger.error(f"AlphaFold query error: {e}")

    return None


# ============================================================
# INPUT PARSING & VALIDATION
# ============================================================

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_sequence(raw: str) -> str:
    """Strips FASTA headers, whitespace, and non-AA characters."""
    lines = raw.strip().splitlines()
    seq_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('>'):
            continue
        seq_lines.append(stripped)
    sequence = ''.join(seq_lines).upper()
    sequence = sequence.replace('*', '').replace('-', '').replace('.', '')
    return sequence

def extract_protein_id_from_header(header: str | None) -> str | None:
    """Extracts protein ID from FASTA header. Multi-format support."""
    if not header: return None
    h = header.strip()
    # UniProt: >sp|P01308|INS_HUMAN or >tr|A0A0J9|...
    m = re.match(r'^>\s*(sp|tr)\|([^|]+)\|', h, flags=re.IGNORECASE)
    if m: return m.group(2)
    # PDB-style: >pdb|1ATP|A or >1ATP_1|Chain A
    m = re.match(r'^>(?:pdb\|)?(\w{4})', h, flags=re.IGNORECASE)
    if m: return m.group(1)
    # Generic: first word after >
    m = re.match(r'^>\s*(\S+)', h)
    if m:
        pid = m.group(1)
        if "_" in pid and len(pid.split("_")[0]) == 4:
            return pid.split("_")[0]
        return pid
    return None

def parse_fasta_text(content: str) -> tuple[str | None, str | None]:
    """Parses FASTA text (from file or paste). Returns (sequence, protein_id)."""
    content = content.strip()
    header = None
    if content.startswith('>'):
        lines = content.splitlines()
        header = lines[0] if lines else None

    sequence = sanitize_sequence(content)

    if len(sequence) < 10:
        raise ValueError(f"Sequence too short: {len(sequence)} residues (minimum 10).")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        current_app.logger.info(f"Sequence truncated from {len(sequence)} to {MAX_SEQUENCE_LENGTH}")
        sequence = sequence[:MAX_SEQUENCE_LENGTH]
    if not VALID_AA_REGEX.match(sequence):
        invalid = set(c for c in sequence if c not in 'ACDEFGHIKLMNPQRSTVWY')
        raise ValueError(f"Invalid amino acids detected: {invalid}. Only the 20 standard AAs are accepted.")

    protein_id = extract_protein_id_from_header(header)
    return sequence, protein_id

def parse_protein_file(file_path: str) -> tuple[str | None, str | None]:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return parse_fasta_text(content)
    except ValueError:
        raise
    except Exception as e:
        current_app.logger.error(f"File parse error: {e}")
        return None, None


# ============================================================
# API ROUTE — /predict
# ============================================================

@api_bp.post('/predict')
def predict():
    try:
        has_file = 'protein_file' in request.files and request.files['protein_file'].filename != ''
        has_sequence = 'sequence' in request.form and request.form['sequence'].strip() != ''

        if not has_file and not has_sequence:
            return jsonify({'success': False, 'error': 'No input. Upload a FASTA file or paste a sequence.'}), 400

        sequence = None
        req_protein_id = None

        # --- Parse Input (file takes precedence) ---
        if has_file:
            file = request.files['protein_file']
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Unsupported format. Use .fasta, .fa, .pdb, or .txt'}), 400

            file_content = file.read()
            file.seek(0)
            file_hash = hashlib.sha256(file_content).hexdigest()[:16]
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{file_hash}_{filename}")
            file.save(file_path)

            try:
                sequence, req_protein_id = parse_protein_file(file_path)
            except ValueError as ve:
                try: os.remove(file_path)
                except OSError: pass
                return jsonify({'success': False, 'error': str(ve)}), 400
            finally:
                try: os.remove(file_path)
                except OSError: pass

        elif has_sequence:
            raw_input = request.form['sequence'].strip()
            try:
                sequence, req_protein_id = parse_fasta_text(raw_input)
            except ValueError as ve:
                return jsonify({'success': False, 'error': str(ve)}), 400
            form_pid = request.form.get('protein_id', '').strip()
            if form_pid:
                req_protein_id = form_pid

        if not sequence:
            return jsonify({'success': False, 'error': 'Sequence extraction failed.'}), 400

        t0 = time.time()

        # --- Conservation Entropy (always computed, single forward pass) ---
        entropy_map = None
        try:
            entropy_map = compute_conservation_entropy(sequence)
        except Exception as e:
            current_app.logger.error(f"Conservation entropy failed: {e}")

        # --- Pseudo-Perplexity (sequence naturalness) ---
        naturalness = None
        try:
            naturalness = compute_pseudo_perplexity(sequence)
        except Exception as e:
            current_app.logger.error(f"Pseudo-perplexity failed: {e}")

        # --- Variant Effect (only if original_sequence provided) ---
        delta_heatmap = None
        original_seq = request.form.get('original_sequence', None)
        if original_seq and sequence:
            original_seq_clean = original_seq.upper().strip()
            if original_seq_clean != sequence:
                try:
                    delta_heatmap = compute_variant_effect(original_seq_clean, sequence)
                except Exception as e:
                    current_app.logger.error(f"Variant effect calculation failed: {e}")

        t_ms = round(time.time() - t0, 4)

        # --- Structural Discovery ---
        p_id = req_protein_id or f"GEN_{hashlib.md5(sequence.encode()).hexdigest()[:8]}"
        structure = None
        try:
            structure = get_structure_url(p_id)
        except Exception as e:
            current_app.logger.error(f"Structure lookup failed: {e}")

        # --- Hotspot Detection ---
        hotspots = []
        if entropy_map:
            for i, h in enumerate(entropy_map):
                if h < ENTROPY_HOTSPOT_THRESHOLD:
                    entry = {'position': i + 1, 'entropy': h, 'residue': sequence[i] if i < len(sequence) else '?'}
                    if delta_heatmap and i < len(delta_heatmap) and delta_heatmap[i] != 0.0:
                        entry['llr'] = delta_heatmap[i]
                        entry['critical'] = True  # Conserved AND mutated
                    hotspots.append(entry)

        return jsonify({
            'success': True,
            '_extracted_sequence': sequence,
            'sequence_length': len(sequence),
            'protein_id': p_id,
            'naturalness': naturalness,
            'entropy_map': entropy_map,
            'delta_heatmap': delta_heatmap,
            'hotspots': hotspots,
            'structure': structure,
            'metrics': {
                'processing_time': t_ms,
                'model': 'esm2_t12_35M_UR50D',
                'mode': 'zero-shot'
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        current_app.logger.exception("API Prediction Error")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
