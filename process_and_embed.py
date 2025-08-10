import pandas as pd
from transformers import AutoTokenizer, EsmModel
import torch
import numpy as np
from tqdm import tqdm # Import tqdm

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
df_merged['Pfam_Family_IDs'].fillna('Unknown', inplace=True) # Updated column name

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"ESM-2 model running on {device}.")
except Exception as e:
    print(f"Error loading ESM-2 model or tokenizer: {e}")
    print("Please ensure 'transformers' and 'torch' are correctly installed and configured.")
    exit()

def get_esm_embeddings_batch(sequences, model, tokenizer, device, max_length=1024):
    # Filter out NaN sequences and store their original indices
    valid_sequences = [s for s in sequences if pd.notna(s)]
    
    if not valid_sequences:
        # If all sequences are NaN, return an array of NaNs with correct shape
        return np.full((len(sequences), model.config.hidden_size), np.nan, dtype=np.float32)

    # Tokenize and move to device
    inputs = tokenizer(valid_sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get embeddings for [CLS] token
    batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

    # Reconstruct full embeddings array with NaNs for original NaN positions
    full_embeddings = np.full((len(sequences), model.config.hidden_size), np.nan, dtype=np.float32)
    valid_idx_counter = 0
    for i, seq in enumerate(sequences):
        if pd.notna(seq):
            full_embeddings[i] = batch_embeddings[valid_idx_counter]
            valid_idx_counter += 1
    return full_embeddings

# Generate embeddings in batches
print("Generating ESM-2 embeddings for sequences in batches...")
all_embeddings = []
batch_size = 32 # You can adjust this batch size based on your GPU memory

# Iterate through the DataFrame in chunks/batches
for i in tqdm(range(0, len(df_merged), batch_size), desc="Generating Embeddings"):
    batch_sequences = df_merged['Sequence'].iloc[i:i+batch_size].tolist()
    
    try:
        batch_embeddings_array = get_esm_embeddings_batch(batch_sequences, model, tokenizer, device)
        all_embeddings.extend(batch_embeddings_array)
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        # If an error occurs in a batch, append NaNs for all sequences in that batch
        for _ in range(len(batch_sequences)):
            all_embeddings.append(np.full(model.config.hidden_size, np.nan, dtype=np.float32))

# Assign the collected embeddings to the DataFrame
df_merged["ESM2_Embedding"] = all_embeddings

print("\n--- ESM-2 Embedding Generation Complete ---")
print("First 5 rows with ESM2_Embedding:")
print(df_merged[['UniProt_ID', 'Sequence', 'ESM2_Embedding']].head())

# You can now save this processed DataFrame if needed
df_merged.to_csv("cleaned_and_embedded_protein_data.tsv", sep='\t', index=False)
print("\nCleaned and embedded data saved to cleaned_and_embedded_protein_data.tsv")
