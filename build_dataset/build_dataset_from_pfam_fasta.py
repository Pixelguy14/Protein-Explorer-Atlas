#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cruza un archivo Pfam (csv/tsv) con un FASTA de UniProt y produce un CSV:
    accession, pfamA_acc, sequence, used_region, seq_start, seq_end, length

- Pfam: columnas mínimas -> pfamseq_acc, pfamA_acc (opcionales: seq_start, seq_end)
- FASTA: headers tipo >sp|A0A437CBF4|... o >A0A437CBF4 ...

Puedes definir rutas aquí abajo, o pasarlas por CLI para sobrescribir.
"""

import csv, argparse, re
from pathlib import Path

# =========== CONFIGURACIÓN POR CÓDIGO ===========
# (Edita estas rutas a las tuyas; USE_MODE: "domain" o "full")
PFAM_FILE   = r".\test_files\pfam_map.csv"
FASTA_FILE  = r".\test_files\sequences.fasta"
OUTPUT_FILE = r".\dataset.csv"
USE_MODE    = "domain"
# ================================================

def sniff_delimiter(p: Path):
    # Lee primera línea con utf-8-sig para tragarse el BOM si existe
    with p.open("r", encoding="utf-8-sig", errors="ignore") as f:
        first = f.readline()
    if "\t" in first and (first.count("\t") >= first.count(",")):
        return "\t"
    return ","

def read_pfam_table(p: Path):
    delim = sniff_delimiter(p)
    rows = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        # Normaliza nombres de columnas (quita BOM y espacios)
        r.fieldnames = [fn.lstrip("\ufeff").strip() for fn in (r.fieldnames or [])]
        need = {"pfamseq_acc", "pfamA_acc"}
        if not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"El archivo Pfam debe incluir columnas: {need}. Encontradas: {r.fieldnames}")
        for row in r:
            # Normaliza claves y valores
            row = {k.lstrip("\ufeff").strip(): (v or "").strip() for k, v in row.items()}
            acc = row.get("pfamseq_acc", "")
            fam = row.get("pfamA_acc", "")
            if not acc or not fam:
                continue
            s, e = row.get("seq_start"), row.get("seq_end")
            seq_start = int(s) if s and s.isdigit() else None
            seq_end   = int(e) if e and e.isdigit() else None
            rows.append({"acc": acc, "fam": fam, "seq_start": seq_start, "seq_end": seq_end})
    return rows

ACC_RE = re.compile(r"^>?([^|>]*\|)?(?P<acc>[A-Z0-9]+)(\|[^ ]*)?")

def parse_fasta(fasta_path: Path):
    """
    Devuelve dict acc -> sequence (A-Z). Extrae accession de:
    >sp|A0A...|..., >tr|..., o >A0A...
    """
    acc2seq = {}
    acc = None
    seq_chunks = []
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if acc and seq_chunks:
                    acc2seq.setdefault(acc, "".join(seq_chunks))
                seq_chunks = []
                m = ACC_RE.match(line[1:])
                acc = m.group("acc") if m else None
            else:
                # Mantén solo letras (aminoácidos). Deja X/B/Z si aparecen.
                seq_chunks.append(re.sub(r"[^A-Za-z]", "", line).upper())
        if acc and seq_chunks:
            acc2seq.setdefault(acc, "".join(seq_chunks))
    return acc2seq

def slice_domain(seq, start, end):
    """
    Pfam usa 1-based e inclusivo. Convertimos a 0-based (fin exclusivo).
    Si faltan límites -> secuencia completa.
    """
    if start is None or end is None:
        return seq, "full"
    i0 = max(0, start - 1)
    i1 = min(len(seq), end)
    if i0 >= i1 or i0 >= len(seq):
        return "", "invalid"
    return seq[i0:i1], "domain"

def main():
    # CLI opcional: sobrescribe los valores definidos por código
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--pfam",  default=PFAM_FILE,   help="CSV/TSV con pfamseq_acc, pfamA_acc, (opc.) seq_start, seq_end")
    ap.add_argument("--fasta", default=FASTA_FILE,  help="FASTA con secuencias UniProt")
    ap.add_argument("--out",   default=OUTPUT_FILE, help="CSV de salida")
    ap.add_argument("--use",   choices=["full","domain"], default=USE_MODE,
                    help="full = secuencia completa; domain = recorte por seq_start/seq_end si existen")
    args = ap.parse_args()

    pfam_path  = Path(args.pfam)
    fasta_path = Path(args.fasta)
    out_path   = Path(args.out)

    if not pfam_path.exists() or not fasta_path.exists():
        raise SystemExit("No encuentro los archivos de entrada.")

    print(f"[INFO] Pfam:  {pfam_path}")
    print(f"[INFO] FASTA: {fasta_path}")
    print(f"[INFO] Modo:  {args.use}")
    print("[INFO] Leyendo Pfam…")
    pf_rows = read_pfam_table(pfam_path)
    print(f"[INFO] Registros Pfam: {len(pf_rows)}")

    print("[INFO] Parseando FASTA…")
    acc2seq = parse_fasta(fasta_path)
    print(f"[INFO] Secuencias en FASTA: {len(acc2seq)}")

    kept, missing_seq, empty_cut = 0, 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["accession","pfamA_acc","sequence","used_region","seq_start","seq_end","length"])
        for r in pf_rows:
            acc, fam = r["acc"], r["fam"]
            seq = acc2seq.get(acc)
            if not seq:
                missing_seq += 1
                continue
            if args.use == "domain":
                cut, region = slice_domain(seq, r["seq_start"], r["seq_end"])
                if region == "invalid" or not cut:
                    empty_cut += 1
                    continue
                out_seq, used = cut, "domain"
            else:
                out_seq, used = seq, "full"
            w.writerow([acc, fam, out_seq, used, r["seq_start"], r["seq_end"], len(out_seq)])
            kept += 1

    print(f"[OK] Escrito: {out_path}")
    print(f"[STATS] Filas guardadas: {kept}")
    print(f"[STATS] Sin secuencia en FASTA: {missing_seq}")
    if args.use == "domain":
        print(f"[STATS] Recorte inválido/vacío: {empty_cut}")

if __name__ == "__main__":
    main()
