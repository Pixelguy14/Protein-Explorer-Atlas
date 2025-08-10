import pandas as pd # type: ignore
from transformers import AutoTokenizer, EsmModel #type: ignore
import torch # type: ignore
import numpy as np

# --- Part 1: Data Cleaning and Filtering ---

# Load the unified dataset
print("Loading unified_protein_data.tsv...")
try:
    df_merged = pd.read_csv("unified_protein_data.tsv", sep='\t')
    print(f"Initial shape: {df_merged.shape}")
except FileNotFoundError:
    print("Error: unified_protein_data.tsv not found. Please ensure data_processing.py has been run.")
    exit()

# 1. Duplicados: Eliminar IDs duplicados conservando la entrada más reciente.
# Assuming 'most recent' means the last occurrence in the current DataFrame order.
initial_rows = df_merged.shape[0]
df_merged.drop_duplicates(subset=['UniProt_ID'], keep='last', inplace=True)
print(f"Removed {initial_rows - df_merged.shape[0]} duplicate UniProt_IDs.")
print(f"Shape after duplicate removal: {df_merged.shape}")

# 2. Secuencias inválidas: Remover secuencias con caracteres no canónicos o longitud < 50 aa.
# Define canonical amino acid characters (20 standard amino acids)
canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")

# Function to check for non-canonical characters
def contains_non_canonical(sequence):
    if pd.isna(sequence):
        return True # Treat NaN sequences as invalid
    return any(char not in canonical_aa for char in sequence.upper())

# Filter out sequences with non-canonical characters
initial_rows = df_merged.shape[0]
df_merged = df_merged[~df_merged['Sequence'].apply(contains_non_canonical)]
print(f"Removed {initial_rows - df_merged.shape[0]} rows with non-canonical amino acids.")
print(f"Shape after non-canonical filter: {df_merged.shape}")

# Filter out sequences with length < 50 aa
initial_rows = df_merged.shape[0]
df_merged = df_merged[df_merged['Sequence'].apply(lambda x: len(x) >= 50 if pd.notna(x) else False)]
print(f"Removed {initial_rows - df_merged.shape[0]} rows with sequence length < 50 aa.")
print(f"Shape after length filter: {df_merged.shape}")

# 3. Datos faltantes:
# Proteínas sin clase: Eliminadas.
# The data_processing.py script fills missing 'Protein class' with 'Unknown'.
# So, we remove rows where 'Protein class' is 'Unknown'.
initial_rows = df_merged.shape[0]
df_merged = df_merged[df_merged['Protein class'] != 'Unknown']
print(f"Removed {initial_rows - df_merged.shape[0]} rows with 'Unknown' Protein class.")
print(f"Shape after 'Protein class' filter: {df_merged.shape}")

# Pfam desconocidos: Etiquetar como Unknown.
# This is already handled by data_processing.py, but we can re-confirm/re-apply if needed.
df_merged['Pfam_Family_ID'].fillna('Unknown', inplace=True)
print("Ensured 'Pfam_Family_ID' missing values are labeled as 'Unknown'.")

print("\n--- Data Cleaning and Filtering Complete ---")
print(f"Final shape after cleaning: {df_merged.shape}")
print(df_merged.head())

# --- Part 2: Generación de Embeddings con ESM-2 ---

print("\n--- Starting ESM-2 Embedding Generation ---")
print("This step can be computationally intensive and time-consuming, especially for large datasets.")
print("It also requires significant RAM and potentially GPU memory.")

# Load ESM-2 tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("ESM-2 model moved to GPU.")
    else:
        print("ESM-2 model running on CPU. Consider using a GPU for faster processing.")
except Exception as e:
    print(f"Error loading ESM-2 model or tokenizer: {e}")
    print("Please ensure 'transformers' and 'torch' are correctly installed and configured.")
    exit()

def get_esm_embedding(sequence):
    if pd.isna(sequence):
        return np.nan # Return NaN for missing sequences
    try:
        # Ensure sequence is a string
        sequence = str(sequence)
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        # Move inputs to GPU if model is on GPU
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        # Return the embedding of the [CLS] token (first token)
        # Detach from GPU and convert to numpy
        return outputs.last_hidden_state[:,0].cpu().numpy().flatten()
    except Exception as e:
        print(f"Error generating embedding for sequence: {sequence[:50]}... Error: {e}")
        return np.nan # Return NaN if embedding generation fails

# Apply the embedding function
# It's recommended to process in batches for very large datasets,
# but for simplicity, we'll apply directly.
print("Generating ESM-2 embeddings for sequences...")
df_merged["ESM2_Embedding"] = df_merged["Sequence"].apply(get_esm_embedding)

print("\n--- ESM-2 Embedding Generation Complete ---")
print("First 5 rows with ESM2_Embedding:")
print(df_merged[['UniProt_ID', 'Sequence', 'ESM2_Embedding']].head())

# You can now save this processed DataFrame if needed
df_merged.to_csv("cleaned_and_embedded_protein_data.tsv", sep='\t', index=False)
print("\nCleaned and embedded data saved to cleaned_and_embedded_protein_data.tsv")
