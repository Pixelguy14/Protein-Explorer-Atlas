
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from tqdm import tqdm

# --- 1. Model and Data Loading ---

print("--- Loading Model and Encoders ---")

# Define the same ProteinCNN class architecture
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, categorical_feature_dim, num_classes):
        super(ProteinCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        cnn_output_dim = 128 * (embedding_dim // 2)

        self.fc1 = nn.Linear(cnn_output_dim + categorical_feature_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_emb, x_cat):
        x_emb = self.conv1(x_emb)
        x_emb = self.relu(x_emb)
        x_emb = self.pool(x_emb)
        x_emb = x_emb.view(x_emb.size(0), -1)

        x = torch.cat((x_emb, x_cat), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load encoders and binarizers
try:
    label_encoder = joblib.load('Training_results/label_encoder.joblib')
    mlb_pfam = joblib.load('Training_results/mlb_pfam.joblib')
    mlb_go = joblib.load('Training_results/mlb_go.joblib')
    print("Loaded label_encoder, mlb_pfam, and mlb_go from 'Training_results/'.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure you have run 'classify_proteins_cnn.py' to train and save the model and encoders.")
    exit()

# Load the data
print("Loading cleaned and embedded data...")
try:
    #df = pd.read_csv("Processed_datasets/cleaned_and_embedded_protein_data.tsv", sep='\t')
    df = pd.read_csv("enriched_protein_data.tsv", sep='\t')
    print("Loaded 'cleaned_and_embedded_protein_data.tsv'.")
except FileNotFoundError:
    print("Error: 'cleaned_and_embedded_protein_data.tsv' not found. Please run 'process_and_embed.py' first.")
    exit()

# --- 2. Prepare Model for Inference ---

# Determine model parameters from loaded objects
num_classes = len(label_encoder.classes_)
categorical_feature_dim = len(mlb_pfam.classes_) + len(mlb_go.classes_)

# A helper function to parse embeddings from string
def parse_embedding(val):
    if isinstance(val, str):
        val = val.strip().replace('\n', '').replace('\r', '')
        if val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        try:
            # First try splitting by space, then by comma
            arr = np.fromstring(val, sep=' ')
            if arr.size == 0:
                arr = np.fromstring(val, sep=',')
            return arr
        except Exception:
            # As a fallback for more complex string formats
            return np.array(eval(val))
    return val

# Get one embedding to determine the embedding_dim
temp_embedding = parse_embedding(df['ESM2_Embedding'].iloc[0])
embedding_dim = len(temp_embedding)

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinCNN(embedding_dim, categorical_feature_dim, num_classes)

# Load the trained model state
try:
    model.load_state_dict(torch.load("Training_results/protein_cnn_model.pth", map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model 'protein_cnn_model.pth' loaded successfully on {device}.")
except FileNotFoundError:
    print("Error: 'Training_results/protein_cnn_model.pth' not found. Please train the model first.")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    print("The model architecture in this script might not match the one that was saved.")
    exit()


# --- 3. Generate Predictions ---

print("\n--- Generating predictions for all proteins ---")
predictions = []

# Helper to split IDs
def split_and_clean_ids(id_string):
    if pd.isna(id_string) or id_string == 'Unknown':
        return []
    return id_string.split(',')

# Use tqdm for progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Predicting"):
    try:
        # 1. Prepare Embedding Tensor
        embedding = parse_embedding(row['ESM2_Embedding'])
        if embedding is None or embedding.size == 0:
            continue
        emb_tensor = torch.tensor(embedding.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        # 2. Prepare Categorical Features Tensor
        pfam_ids = split_and_clean_ids(row['Pfam_Family_IDs'])
        go_ids = split_and_clean_ids(row['GO_IDs'])

        pfam_encoded = mlb_pfam.transform([pfam_ids])
        go_encoded = mlb_go.transform([go_ids])

        cat_combined = np.hstack((pfam_encoded, go_encoded)).astype(np.float32)
        cat_tensor = torch.tensor(cat_combined).to(device)

        # 3. Predict
        with torch.no_grad():
            outputs = model(emb_tensor, cat_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]

        # 4. Store result
        predictions.append({
            "protein_id": row['UniProt_ID'],
            "sequence": row['Sequence'],
            "predicted_class": predicted_class,
            "confidence": round(confidence.item(), 4),
            "true_class": row.get('Protein class', 'N/A'), # Include true class if available
            "pdb_ids": row.get('PDB_IDs', 'Null'),
            "pfam_names": row.get('Pfam_Names', 'Null')
        })
    except Exception as e:
        print(f"Could not process row {index} ({row['UniProt_ID']}): {e}")


# --- 4. Save Predictions ---

output_file = 'predictions.json'
with open(output_file, 'w') as f:
    json.dump(predictions, f, indent=4)

print(f"\n--- Predictions saved to {output_file} ---")
print(f"Successfully generated predictions for {len(predictions)} proteins.")
