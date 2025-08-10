import os, sys, time, csv, math, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

# =======================
# Backend: DML -> CPU fallback
# =======================
def pick_device():
    forced = os.environ.get("BACKEND", "").lower().strip()  # "cpu" | "dml" | ""
    if forced == "cpu":
        print("[INFO] Forzando backend CPU por BACKEND=cpu")
        return torch.device("cpu"), "cpu"
    try:
        import torch_directml
        dml = torch_directml.device()
        _ = torch.tensor([1.0], device=dml)
        if forced == "dml":
            print("[INFO] Forzando backend DirectML por BACKEND=dml")
        else:
            print("[INFO] Backend elegido automáticamente: DirectML")
        return dml, "directml"
    except Exception as e:
        if forced == "dml":
            print("[WARN] BACKEND=dml solicitado pero DirectML no está disponible:", repr(e))
        print("[INFO] Usando CPU (fallback).")
        return torch.device("cpu"), "cpu"

DEVICE, BACKEND = pick_device()

# =======================
# Modelo / IO
# =======================
MODEL_ID = "facebook/esm2_t6_8M_UR50D"
# clave para VRAM: bajar L y batch
MAX_LEN = 512                 # <= 512 ayuda muchísimo con L^2
INIT_BATCH_SIZE = 2           # baja a 1 si aún revienta
MIN_BATCH_SIZE = 1
WINDOW = 512                  # tamaño de ventana para seq largas
STRIDE = 448                  # solape (512-448=64 aa)
INPUT_TSV  = "test_files\dataset_limpio.tsv"
OUTPUT_CSV = "test_files\esm2_8M_embeddings.csv"
FLUSH_EVERY = 200             # escribe a disco cada N batches

print("[INFO] Cargando tokenizer y modelo:", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModel.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()

def mean_pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

def clean_seq(seq: str) -> str:
    import re
    return re.sub(r"[^A-Za-z]", "", (seq or "")).upper()

def read_tsv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        r.fieldnames = [h.lstrip("\ufeff").strip() for h in (r.fieldnames or [])]
        required = {"protein_id", "sequence", "family_functional"}
        if not required.issubset(set(r.fieldnames)):
            raise SystemExit(f"El TSV debe incluir columnas {required}. Encontradas: {r.fieldnames}")
        for row in r:
            pid = (row.get("protein_id") or "").strip()
            seq = clean_seq(row.get("sequence") or "")
            fam = (row.get("family_functional") or "").strip()
            if not pid or not seq or not fam:
                continue
            rows.append((pid, seq, fam))
    return rows

def window_sequences(seq: str, win=WINDOW, stride=STRIDE):
    if len(seq) <= win:
        return [seq]
    chunks = []
    i = 0
    while i < len(seq):
        chunk = seq[i:i+win]
        if not chunk:
            break
        chunks.append(chunk)
        if i + win >= len(seq):
            break
        i += stride
    return chunks

def embed_batch(seqs, device):
    enc = tokenizer(
        seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        last_hidden = out.last_hidden_state
        emb = mean_pool_hidden(last_hidden, enc["attention_mask"])
        return emb.detach().cpu().numpy()

def safe_embed_batch(seqs):
    # intenta en DML; si OOM, reintenta en CPU para este batch
    try:
        return embed_batch(seqs, DEVICE)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("out of memory" in msg) or ("oom" in msg) or ("e_outofmemory" in msg):
            print("[WARN] OOM en backend", BACKEND, "para este batch. Reintentando en CPU...")
            try:
                return embed_batch(seqs, torch.device("cpu"))
            except Exception as e2:
                print("[ERROR] Falló también en CPU:", repr(e2))
                raise
        raise

def run_inference(records):
    """
    records: lista (protein_id, sequence, family)
    Estrategia:
      - Ordenar por longitud ascendente (menos padding).
      - Batching adaptativo (reduce si hay OOM).
      - Ventaneo para secuencias largas y promedio de ventanas.
    """
    # Ordena por longitud para minimizar padding
    records = sorted(records, key=lambda x: len(x[1]))
    bs = INIT_BATCH_SIZE
    total_rows = len(records)

    # Prepara salida incremental
    header_written = False
    processed = 0
    batch_index = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = None
        pbar = tqdm(total=total_rows, desc="Procesando secuencias", unit="seq")

        i = 0
        while i < len(records):
            try:
                chunk = records[i:i+bs]
                batch_index += 1

                # Prepara secuencias con ventaneo
                expanded = []   # (pid, fam, L_orig, [subseqs])
                for pid, seq, fam in chunk:
                    subs = window_sequences(seq, WINDOW, STRIDE)
                    expanded.append((pid, fam, len(seq), subs))

                # Embedding por ventanas y promedio
                all_rows = []
                for pid, fam, Lorig, subs in expanded:
                    vecs = []
                    # procesamos las sub-secuencias en mini-lotes para evitar OOM
                    mini_bs = max(1, min(len(subs), 8))
                    j = 0
                    while j < len(subs):
                        mini = subs[j:j+mini_bs]
                        vecs.append(safe_embed_batch(mini))
                        j += mini_bs
                    V = np.vstack(vecs)            # [n_windows, H]
                    vmean = V.mean(axis=0)         # [H]
                    all_rows.append((pid, fam, Lorig, vmean))

                # Inicializa writer con dimensión H
                if not header_written and all_rows:
                    H = all_rows[0][3].shape[0]
                    header = ["protein_id", "family_functional", "seq_len"] + [f"dim_{k}" for k in range(H)]
                    writer = csv.writer(f)
                    writer.writerow(header)
                    header_written = True

                # Escribe filas
                for pid, fam, Lorig, v in all_rows:
                    writer.writerow([pid, fam, Lorig] + v.tolist())
                processed += len(chunk)

                pbar.update(processed)

                # Avanza ventana principal
                i += bs

                # Flush periódico
                if batch_index % FLUSH_EVERY == 0:
                    f.flush()
                    os.fsync(f.fileno())
                    print(f"[INFO] Progreso: {processed}/{len(records)}")

            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg) or ("oom" in msg) or ("e_outofmemory" in msg):
                    if bs > MIN_BATCH_SIZE:
                        new_bs = max(MIN_BATCH_SIZE, bs // 2)
                        print(f"[WARN] OOM con batch_size={bs}. Reintentando con batch_size={new_bs}...")
                        bs = new_bs
                        time.sleep(0.5)
                        continue
                    else:
                        print("[ERROR] OOM incluso con batch_size=1. Aborto.")
                        raise
                else:
                    raise
        pbar.close()

    print(f"[OK] Guardado embeddings en {OUTPUT_CSV}")

# ================ MAIN =================
data = read_tsv(INPUT_TSV)
print(f"[INFO] Filas a procesar: {len(data)} | Backend: {BACKEND}")
run_inference(data)
print("[DONE]")
