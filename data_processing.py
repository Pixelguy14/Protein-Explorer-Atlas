import pandas as pd # type: ignore
from Bio import SeqIO # type: ignore

def parse_fasta(fasta_file):
    """
    Parses a FASTA file and extracts UniProt IDs and sequences.
    """
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract UniProt ID from the header (e.g., sp|A0A087X0M5|TVB18_HUMAN)
        uniprot_id = record.id.split('|')[1]
        sequences[uniprot_id] = str(record.seq)
    return pd.DataFrame(list(sequences.items()), columns=['UniProt_ID', 'Sequence'])

def parse_pfam_regions(pfam_regions_file):
    """
    Parses the Pfam-A.regions.tsv file to extract Pfam family information.
    """
    # Assuming tab-separated and relevant columns are 'pfamseq_acc' and 'pfamA_acc'
    df = pd.read_csv(pfam_regions_file, sep='\t', usecols=['pfamseq_acc', 'pfamA_acc'], engine='c')
    df.rename(columns={'pfamseq_acc': 'UniProt_ID', 'pfamA_acc': 'Pfam_Family_ID'}, inplace=True)
    return df

def parse_protein_atlas(protein_atlas_file):
    """
    Parses the proteinatlas.tsv file to extract relevant annotations.
    """
    # Assuming tab-separated and relevant columns are 'Uniprot' and 'Protein class', 'Biological process', 'Molecular function'
    df = pd.read_csv(protein_atlas_file, sep='\t', usecols=['Uniprot', 'Protein class', 'Biological process', 'Molecular function'], engine='c')
    df.rename(columns={'Uniprot': 'UniProt_ID'}, inplace=True)
    return df

if __name__ == "__main__":
    fasta_file = "/home/pixel/Downloads/Hackaton/Protein-Explorer-Atlas/data/UP000005640_9606.fasta"
    pfam_regions_file = "/home/pixel/Downloads/Hackaton/Protein-Explorer-Atlas/data/Pfam-A.regions.tsv"
    protein_atlas_file = "/home/pixel/Downloads/Hackaton/Protein-Explorer-Atlas/data/proteinatlas.tsv"

    print("Parsing FASTA file...")
    fasta_df = parse_fasta(fasta_file)
    print(f"FASTA data shape: {fasta_df.shape}")
    print(fasta_df.head())

    print("\nParsing Pfam regions file...")
    pfam_df = parse_pfam_regions(pfam_regions_file)
    print(f"Pfam data shape: {pfam_df.shape}")
    print(pfam_df.head())

    print("\nParsing Protein Atlas file...")
    hpa_df = parse_protein_atlas(protein_atlas_file)
    print(f"HPA data shape: {hpa_df.shape}")
    print(hpa_df.head())

    # Merge dataframes
    print("\nMerging dataframes...")
    # Start with FASTA data as the base
    unified_df = fasta_df

    # Merge with Pfam data
    # A protein can have multiple Pfam domains, so we'll merge based on UniProt_ID
    # and keep all Pfam families associated with a protein.
    unified_df = pd.merge(unified_df, pfam_df, on='UniProt_ID', how='left')

    # Merge with HPA data
    # HPA data also has UniProt_ID.
    unified_df = pd.merge(unified_df, hpa_df, on='UniProt_ID', how='left')

    print(f"Unified data shape: {unified_df.shape}")
    print(unified_df.head())

    # Data Cleaning and Filtering (initial steps)
    # Remove duplicates (based on UniProt_ID and Sequence)
    initial_rows = unified_df.shape[0]
    unified_df.drop_duplicates(subset=['UniProt_ID', 'Sequence'], inplace=True)
    print(f"Removed {initial_rows - unified_df.shape[0]} duplicate protein sequences.")

    # Filter proteins without Pfam classification (if desired, for now, keep them but note NaNs)
    # unified_df.dropna(subset=['Pfam_Family_ID'], inplace=True)
    # print(f"Filtered out proteins without Pfam classification. New shape: {unified_df.shape}")

    # Handle missing values (e.g., fill with 'Unknown' or specific placeholder)
    unified_df.fillna({'Pfam_Family_ID': 'Unknown', 'Protein class': 'Unknown', 'Biological process': 'Unknown', 'Molecular function': 'Unknown'}, inplace=True)

    print("\nUnified and partially cleaned data:")
    print(unified_df.head())
    print(unified_df.info())

    # Save the unified dataframe (optional, for later use)
    unified_df.to_csv("unified_protein_data.tsv", sep='\t', index=False)
    print("\nUnified data saved to unified_protein_data.tsv")
