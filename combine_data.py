import csv
import json
import os

# --- 1. Parse Pfam-A.clans.tsv (Pfam ID to Name mapping) ---
pfam_names_map = {}
try:
    with open("Pfam-A.clans.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                pfam_acc = row[0].strip()
                pfam_name = row[2].strip()
                pfam_names_map[pfam_acc] = pfam_name
except FileNotFoundError:
    print("Error: Pfam-A.clans.tsv not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"Error parsing Pfam-A.clans.tsv: {e}")
    exit()

# --- 2. Parse pdb_chain_uniprot.csv (UniProt ID to PDB ID mapping) ---
uniprot_pdb_map = {}
try:
    with open("pdb_chain_uniprot.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f) # Default delimiter is comma
        next(reader, None) # Skip header row
        for row in reader:
            if len(row) >= 3:
                uniprot_id = row[2].strip()
                pdb_id = row[0].strip()
                if uniprot_id not in uniprot_pdb_map:
                    uniprot_pdb_map[uniprot_id] = set() # Use set to avoid duplicate PDBs
                uniprot_pdb_map[uniprot_id].add(pdb_id)
except FileNotFoundError:
    print("Error: pdb_chain_uniprot.csv not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"Error parsing pdb_chain_uniprot.csv: {e}")
    exit()

# Convert sets to lists for JSON serialization later
for key in uniprot_pdb_map:
    uniprot_pdb_map[key] = list(uniprot_pdb_map[key])

# --- 3. Combine Data and Create New TSV ---
input_cleaned_data_file = "Processed_datasets/cleaned_and_embedded_protein_data.tsv"
output_enriched_data_file = "enriched_protein_data.tsv"

try:
    with open(input_cleaned_data_file, 'r', encoding='utf-8') as infile, \
         open(output_enriched_data_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        header = next(reader)
        new_header = header + ["PDB_IDs", "Pfam_Names"]
        writer.writerow(new_header)

        uniprot_id_idx = header.index("UniProt_ID")
        pfam_family_ids_idx = header.index("Pfam_Family_IDs")

        for row in reader:
            if len(row) < len(header):
                print(f"Skipping malformed row: {row}")
                continue

            uniprot_id = row[uniprot_id_idx].strip()
            original_pfam_ids_str = row[pfam_family_ids_idx].strip()

            pdb_ids = uniprot_pdb_map.get(uniprot_id, [])
            pdb_ids_str = ";".join(pdb_ids)

            pfam_names = []
            if original_pfam_ids_str:
                for pfam_id in original_pfam_ids_str.split(';'):
                    pfam_id = pfam_id.strip()
                    if pfam_id in pfam_names_map:
                        pfam_names.append(pfam_names_map[pfam_id])
            pfam_names_str = ";".join(pfam_names)

            new_row = row + [pdb_ids_str, pfam_names_str]
            writer.writerow(new_row)

    print(f"Successfully created {output_enriched_data_file}")

except FileNotFoundError:
    print(f"Error: One of the input files not found. Please ensure {input_cleaned_data_file}, Pfam-A.clans.tsv, and pdb_chain_uniprot.csv are in the current directory.")
    exit()
except Exception as e:
    print(f"Error combining data: {e}")
    exit()
