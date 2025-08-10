import os, csv, time, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =======================
# CONFIG
# =======================
MODEL_ID = os.getenv("ESM2_MODEL", "facebook/esm2_t12_35M_UR50D")  # 35M va bien en 8GB; usa t6_8M o t30_150M si quieres
INPUT_TSV  = os.getenv("INPUT_TSV", "dataset_limpio.tsv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "esm2_cuda_embeddings.csv")

MAX_LEN = 1022         # límite práctico de ESM-2
INIT_BATCH = 32        # sube/baja según VRAM (RTX 4060 8GB suele aguantar 32-48 con 35M)
MIN_BATCH  = 1
SORT_BY_LEN = True

# =======================
# CUDA knobs
# =======================
assert torch.cuda.is_available(), "No hay GPU CUDA disponible"
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 acelera en Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =======================
# Utilidades
# =======================
AA_RE = re.compile(r"[^A-Za-z]")

def clean_seq(s: str) -> str:
    return AA_RE.sub("", (s or "")).upper()

def read_tsv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        r.fieldnames = [h.lstrip("\ufeff").strip() for h in (r.fieldnames or [])]
        need = {"protein_id", "sequence", "family_functional"}
        if not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"El TSV debe incluir columnas {need}. Encontradas: {r.fieldnames}")
        for row in r:
            pid = (row.get("protein_id") or "").strip()
            seq = clean_seq(row.get("sequence") or "")
            fam = (row.get("family_functional") or "").strip()
            if pid and seq and fam:
                rows.append((pid, seq, fam))
    return rows

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

# =======================
# Carga modelo/tokenizer
# =======================
print(f"[INFO] Modelo: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModel.from_pretrained(MODEL_ID)
# bajar a FP16 para reducir VRAM
model.half().to(device)
model.eval()

# =======================
# Datos
# =======================
data = read_tsv(INPUT_TSV)  # (protein_id, seq, fam)
if not data:
    raise SystemExit("No se leyeron filas válidas del TSV.")
if SORT_BY_LEN:
    data = sorted(data, key=lambda x: len(x[1]))
N = len(data)
print(f"[INFO] Filas a procesar: {N}")

# =======================
# Inference
# =======================
def embed_batch(seqs):
    enc = tokenizer(
        seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
    )
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        out = model(**enc)
        last = out.last_hidden_state          # [B, L, H]
        pooled = mean_pool(last, enc["attention_mask"])  # [B, H]
    return pooled.detach().cpu().numpy()

bs = INIT_BATCH
header_written = False
processed = 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = None
    pbar = tqdm(total=N, desc="Embeddings", unit="seq")

    i = 0
    while i < N:
        try:
            chunk = data[i:i+bs]
            seqs = [s for _, s, _ in chunk]
            vec = embed_batch(seqs)          # [B, H]

            if not header_written:
                H = vec.shape[1]
                header = ["protein_id", "family_functional", "seq_len"] + [f"dim_{k}" for k in range(H)]
                writer = csv.writer(f)
                writer.writerow(header)
                header_written = True

            for (pid, seq, fam), v in zip(chunk, vec):
                writer.writerow([pid, fam, len(seq)] + v.tolist())

            processed += len(chunk)
            pbar.update(len(chunk))
            i += bs

        except RuntimeError as e:
            # CUDA OOM -> reduce batch y reintenta
            if "out of memory" in str(e).lower():
                if bs > MIN_BATCH:
                    new_bs = max(MIN_BATCH, bs // 2)
                    print(f"\n[WARN] CUDA OOM con batch={bs}. Probando batch={new_bs}...")
                    bs = new_bs
                    torch.cuda.empty_cache()
                    time.sleep(0.2)
                    continue
                else:
                    print("[ERROR] OOM incluso con batch=1.")
                    raise
            else:
                raise

    pbar.close()

print(f"[OK] Guardado: {OUTPUT_CSV}")
