import pandas as pd
from Bio import SeqIO

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
    Parses the Pfam-A.regions_human.tsv file to extract Pfam family information
    and aggregates multiple Pfam IDs per UniProt ID.
    """
    # Assuming tab-separated and relevant columns are 'pfamseq_acc' and 'pfamA_acc'
    df = pd.read_csv(pfam_regions_file, sep='\t', usecols=['pfamseq_acc', 'pfamA_acc'], engine='c')
    df.rename(columns={'pfamseq_acc': 'UniProt_ID', 'pfamA_acc': 'Pfam_Family_ID'}, inplace=True)

    # Aggregate multiple Pfam IDs for each UniProt_ID into a single string
    df_agg = df.groupby('UniProt_ID')['Pfam_Family_ID'].apply(lambda x: ','.join(x.astype(str).unique())).reset_index()
    df_agg.rename(columns={'Pfam_Family_ID': 'Pfam_Family_IDs'}, inplace=True) # Rename column to reflect aggregation
    return df_agg

def parse_protein_atlas(protein_atlas_file):
    """
    Parses the proteinatlas.tsv file to extract relevant annotations.
    """
    # Assuming tab-separated and relevant columns are 'Uniprot' and 'Protein class', 'Biological process', 'Molecular function'
    df = pd.read_csv(protein_atlas_file, sep='\t', usecols=['Uniprot', 'Protein class', 'Biological process', 'Molecular function'], engine='c')
    df.rename(columns={'Uniprot': 'UniProt_ID'}, inplace=True)
    return df

def parse_go_annotations(go_annotations_file):
    """
    Parses the goa_human.gaf file to extract UniProt to GO term mappings
    and aggregates multiple GO IDs per UniProt ID.
    """
    # GAF format is tab-separated. We need UniProt ID (col 2) and GO ID (col 5).
    # Skip comment lines starting with '!'
    df = pd.read_csv(go_annotations_file, sep='\t', header=None, comment='!',
                     usecols=[1, 4], names=['UniProt_ID', 'GO_ID'], engine='c')

    # Aggregate multiple GO IDs for each UniProt_ID into a single string
    df_agg = df.groupby('UniProt_ID')['GO_ID'].apply(lambda x: ','.join(x.astype(str).unique())).reset_index()
    df_agg.rename(columns={'GO_ID': 'GO_IDs'}, inplace=True) # Rename column to reflect aggregation
    return df_agg


if __name__ == "__main__":
    # Updated file paths to reflect new subdirectory structure
    fasta_file = "data/UP000005640_9606.fasta"
    pfam_regions_file = "data/pfam/Pfam-A.regions.tsv"
    protein_atlas_file = "data/hpa/proteinatlas.tsv"
    go_annotations_file = "data/goa/goa_human.gaf" # New file path

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

    print("\nParsing GO annotations...")
    go_df = parse_go_annotations(go_annotations_file)
    print(f"GO annotations shape: {go_df.shape}")
    print(go_df.head())

    # Merge dataframes
    print("\nMerging dataframes...")
    # Start with FASTA data as the base
    unified_df = fasta_df

    # Merge with Pfam data (now aggregated)
    unified_df = pd.merge(unified_df, pfam_df, on='UniProt_ID', how='left')

    # Merge with HPA data
    unified_df = pd.merge(unified_df, hpa_df, on='UniProt_ID', how='left')

    # Merge with GO annotations (now aggregated)
    unified_df = pd.merge(unified_df, go_df, on='UniProt_ID', how='left')


    print(f"Unified data shape: {unified_df.shape}")
    print(unified_df.head())

    # Data Cleaning and Filtering (initial steps)
    # Remove duplicates (based on UniProt_ID and Sequence)
    initial_rows = unified_df.shape[0]
    unified_df.drop_duplicates(subset=['UniProt_ID', 'Sequence'], inplace=True)
    print(f"Removed {initial_rows - unified_df.shape[0]} duplicate protein sequences.")

    # Handle missing values (e.g., fill with 'Unknown' or specific placeholder)
    unified_df.fillna({
        'Pfam_Family_IDs': 'Unknown',
        'Protein class': 'Unknown',
        'Biological process': 'Unknown',
        'Molecular function': 'Unknown',
        'GO_IDs': 'Unknown' # New: fill missing GO IDs
    }, inplace=True)

    print("\nUnified and partially cleaned data:")
    print(unified_df.head())
    print(unified_df.info())

    # Save the unified dataframe (optional, for later use)
    unified_df.to_csv("unified_protein_data.tsv", sep='\t', index=False)
    print("\nUnified data saved to unified_protein_data.tsv")