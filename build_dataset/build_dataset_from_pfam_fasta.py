#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cruza un archivo Pfam (csv/tsv) con un FASTA de UniProt y produce un CSV:
    accession, pfamA_acc, sequence, used_region, seq_start, seq_end

- Requiere en el archivo Pfam las columnas: pfamseq_acc, pfamA_acc
  (opcionales: seq_start, seq_end para recortar al dominio).
- FASTA: cabeceras tipo >sp|A0A437CBF4|... o >A0A437CBF4 ...

Uso:
    python build_dataset_from_pfam_fasta.py --pfam [Nombre de archivo no fasta].csv --fasta [Nombre de archivo fasta].fasta --out dataset.csv --use domain || Este es el importante.
    python build_dataset_from_pfam_fasta.py --pfam pfam_map.csv --fasta sequences.fasta --out dataset_full.csv --use full
"""
import csv, argparse, sys, re
from pathlib import Path

def sniff_delimiter(p: Path):
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
            # Normaliza claves del row (por si el BOM vino en los datos)
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
    Devuelve dict acc -> sequence (solo letras A-Z).
    Intenta extraer la accession del header: >sp|A0A...|..., >tr|..., o >A0A...
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
                seq_chunks.append(re.sub(r"[^A-Za-z]", "", line).upper())
        if acc and seq_chunks:
            acc2seq.setdefault(acc, "".join(seq_chunks))
    return acc2seq

def slice_domain(seq, start, end):
    """
    Pfam suele indexar 1-based e inclusivo. Convertimos a 0-based Python.
    Si faltan límites, regresa secuencia completa.
    """
    if start is None or end is None:
        return seq, "full"
    i0 = max(0, start - 1)
    i1 = min(len(seq), end)
    if i0 >= i1 or i0 >= len(seq):
        return "", "invalid"
    return seq[i0:i1], "domain"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfam", required=True, help="CSV/TSV con columnas pfamseq_acc, pfamA_acc, (opcionales) seq_start, seq_end")
    ap.add_argument("--fasta", required=True, help="FASTA con secuencias UniProt")
    ap.add_argument("--out", default="dataset.csv", help="CSV de salida")
    ap.add_argument("--use", choices=["full", "domain"], default="domain",
                    help="full = usar secuencia completa; domain = recortar por seq_start/seq_end si existen")
    args = ap.parse_args()

    pfam_path = Path(args.pfam)
    fasta_path = Path(args.fasta)
    if not pfam_path.exists() or not fasta_path.exists():
        raise SystemExit("No encuentro los archivos de entrada.")

    print("[INFO] Leyendo Pfam…")
    pf_rows = read_pfam_table(pfam_path)
    print(f"[INFO] Registros Pfam: {len(pf_rows)}")

    print("[INFO] Parseando FASTA…")
    acc2seq = parse_fasta(fasta_path)
    print(f"[INFO] Secuencias en FASTA: {len(acc2seq)}")

    kept, missing_seq, empty_cut = 0, 0, 0
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
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

    print(f"[OK] Escrito: {args.out}")
    print(f"[STATS] Filas guardadas: {kept}")
    print(f"[STATS] Sin secuencia en FASTA: {missing_seq}")
    if args.use == "domain":
        print(f"[STATS] Recorte inválido/vacío: {empty_cut}")

if __name__ == "__main__":
    main()
