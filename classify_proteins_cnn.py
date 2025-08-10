import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer # Added MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# --- Part 3: Protein Classification with CNN ---

print("\n--- Starting Protein Classification with CNN ---")

# Load the processed data (assuming it's saved from previous steps)
try:
    df_processed = pd.read_csv("Processed_datasets/cleaned_and_embedded_protein_data.tsv", sep='\t')
    print("Loaded cleaned and embedded data from 'cleaned_and_embedded_protein_data.tsv'.")
except FileNotFoundError:
    print("Error: 'cleaned_and_embedded_protein_data.tsv' not found. Please ensure you saved the output from 'process_and_embed.py'.")
    print("Creating a dummy DataFrame for demonstration purposes. This will not use actual embeddings.")
    # Create a dummy DataFrame for demonstration if the file isn't found
    data = {
        'UniProt_ID': ['P12345', 'Q67890', 'R09876', 'S54321', 'T11223', 'U12345', 'V67890', 'W09876', 'X54321', 'Y11223'],
        'Sequence': ['AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT',
                     'CDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'GHIJKLMNOPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'IJKLMNOPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'KLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT',
                     'CDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'GHIJKLMNOPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'IJKLMNOPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
                     'KLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY'],
        'Pfam_Family_IDs': ['PF00001,PF00002', 'PF00002', 'PF00001', 'PF00003', 'Unknown', 'PF00001', 'PF00002', 'PF00001', 'PF00003', 'Unknown'],
        'Protein class': ['Enzyme', 'Transporter', 'Enzyme', 'Structural', 'Enzyme', 'Transporter', 'Enzyme', 'Structural', 'Enzyme', 'Transporter'],
        'Biological process': ['Metabolism', 'Transport', 'Signaling', 'Structure', 'Metabolism', 'Transport', 'Signaling', 'Structure', 'Metabolism', 'Transport'],
        'Molecular function': ['Catalytic', 'Binding', 'Catalytic', 'Structural', 'Binding', 'Catalytic', 'Binding', 'Catalytic', 'Structural', 'Binding'],
        'GO_IDs': ['GO:0003824,GO:0008152', 'GO:0006810', 'GO:0007165', 'GO:0005198', 'Unknown', 'GO:0003824', 'GO:0006810', 'GO:0007165', 'GO:0005198', 'Unknown'],
        'ESM2_Embedding': [np.random.rand(320) for _ in range(10)]
    }
    df_processed = pd.DataFrame(data)
    print("Dummy DataFrame created.")

# Prepare Features (X) and Target (y)

# Convert embeddings from string to numpy array if necessary
def parse_embedding(val):
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        try:
            arr = np.fromstring(val, sep=' ')
            if arr.size == 0:
                arr = np.fromstring(val, sep=',')
            return arr
        except Exception:
            return np.array(eval(val))
    return val

embeddings = [parse_embedding(e) for e in df_processed['ESM2_Embedding'].values]
X_embeddings = np.vstack(embeddings).astype(np.float32)
X_embeddings_tensor = torch.tensor(X_embeddings).unsqueeze(1)

def split_and_clean_ids(id_string):
    if pd.isna(id_string) or id_string == 'Unknown':
        return []
    return id_string.split(',')

df_processed['Pfam_Family_IDs_list'] = df_processed['Pfam_Family_IDs'].apply(split_and_clean_ids)
df_processed['GO_IDs_list'] = df_processed['GO_IDs'].apply(split_and_clean_ids)

mlb_pfam = MultiLabelBinarizer()
X_pfam_encoded = mlb_pfam.fit_transform(df_processed['Pfam_Family_IDs_list'])
print(f"Pfam features shape: {X_pfam_encoded.shape}")

mlb_go = MultiLabelBinarizer()
X_go_encoded = mlb_go.fit_transform(df_processed['GO_IDs_list'])
print(f"GO features shape: {X_go_encoded.shape}")

X_categorical_combined = np.hstack((X_pfam_encoded, X_go_encoded)).astype(np.float32)
print(f"Combined categorical features shape: {X_categorical_combined.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_processed['Protein class'])
y = torch.tensor(y, dtype=torch.long)

num_classes = len(label_encoder.classes_)
embedding_dim = X_embeddings_tensor.shape[2]
categorical_feature_dim = X_categorical_combined.shape[1]

print(f"Shape of ESM2 features for CNN: {X_embeddings_tensor.shape} (batch_size, channels, embedding_dim)")
print(f"Shape of combined categorical features: {X_categorical_combined.shape}")
print(f"Shape of target (y): {y.shape}")
print(f"Number of classes: {num_classes}")
print(f"ESM2 Embedding dimension: {embedding_dim}")
print(f"Categorical Feature dimension: {categorical_feature_dim}")


# Filtrar clases con solo 1 muestra antes de dividir
unique, counts = np.unique(y.numpy(), return_counts=True)
valid_classes = unique[counts > 1]
valid_idx = np.isin(y.numpy(), valid_classes)
X_embeddings_tensor_filtered = X_embeddings_tensor[valid_idx]
X_categorical_combined_filtered = X_categorical_combined[valid_idx]
y_filtered = y[valid_idx]

X_train_emb, X_test_emb, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_embeddings_tensor_filtered, X_categorical_combined_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

train_dataset = TensorDataset(X_train_emb, torch.tensor(X_train_cat), y_train)
test_dataset = TensorDataset(X_test_emb, torch.tensor(X_test_cat), y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ProteinCNN(embedding_dim, categorical_feature_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
print(f"\nTraining CNN for {num_epochs} epochs...")
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs_emb, inputs_cat, labels in train_loader:
        inputs_emb, inputs_cat, labels = inputs_emb.to(device), inputs_cat.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs_emb, inputs_cat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs_emb.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("\nCNN Training Complete.")

model_save_path = "Training_results/protein_cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

import joblib
print("\nSaving encoders and binarizers...")
joblib.dump(label_encoder, 'Training_results/label_encoder.joblib')
joblib.dump(mlb_pfam, 'Training_results/mlb_pfam.joblib')
joblib.dump(mlb_go, 'Training_results/mlb_go.joblib')
print("Encoders and binarizers saved to 'Training_results/'.")

print("\nEvaluating the CNN model...")
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs_emb, inputs_cat, labels in test_loader:
        inputs_emb, inputs_cat, labels = inputs_emb.to(device), inputs_cat.to(device), labels.to(device)
        outputs = model(inputs_emb, inputs_cat)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

y_true_labels = label_encoder.inverse_transform(y_true)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print(f"Accuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# --- Visualización de métricas ---
from sklearn.metrics import precision_recall_fscore_support


# Definir top_classes (Top 30 clases más frecuentes en y_true_labels)
from collections import Counter
top_n = 30
class_counts = Counter(y_true_labels)
top_classes = [cls for cls, _ in class_counts.most_common(top_n)]

# Obtener métricas por clase
precision, recall, f1, support = precision_recall_fscore_support(y_true_labels, y_pred_labels, labels=top_classes, zero_division=0)

# Graficar clasificación (soporte por clase)
plt.figure(figsize=(12, 6))
plt.bar(top_classes, support)
plt.xticks(rotation=90)
plt.title('Soporte por clase (Top 30)')
plt.xlabel('Clase')
plt.ylabel('Número de muestras')
plt.tight_layout()
plt.savefig('support_per_class.png')
plt.close()
print('Imagen guardada: support_per_class.png')

# Graficar precisión por clase
plt.figure(figsize=(12, 6))
plt.bar(top_classes, precision)
plt.xticks(rotation=90)
plt.title('Precisión por clase (Top 30)')
plt.xlabel('Clase')
plt.ylabel('Precisión')
plt.tight_layout()
plt.savefig('precision_per_class.png')
plt.close()
print('Imagen guardada: precision_per_class.png')

# Graficar recall por clase
plt.figure(figsize=(12, 6))
plt.bar(top_classes, recall)
plt.xticks(rotation=90)
plt.title('Recall por clase (Top 30)')
plt.xlabel('Clase')
plt.ylabel('Recall')
plt.tight_layout()
plt.savefig('recall_per_class.png')
plt.close()
print('Imagen guardada: recall_per_class.png')

# Graficar F1-score por clase
plt.figure(figsize=(12, 6))
plt.bar(top_classes, f1)
plt.xticks(rotation=90)
plt.title('F1-score por clase (Top 30)')
plt.xlabel('Clase')
plt.ylabel('F1-score')
plt.tight_layout()
plt.savefig('f1_per_class.png')
plt.close()
print('Imagen guardada: f1_per_class.png')

# Graficar macro/micro promedios
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

plt.figure(figsize=(8, 6))
plt.bar(['Macro Precision', 'Macro Recall', 'Macro F1'], [macro_precision, macro_recall, macro_f1])
plt.title('Macro Promedios de Métricas (Top 30)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('macro_metrics.png')
plt.close()
print('Imagen guardada: macro_metrics.png')

print("\n--- Protein Classification with CNN Complete ---")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()
print("\nTraining loss plot saved to training_loss.png")

cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)

# Mostrar solo las 30 clases más frecuentes para evitar sobreposición
from collections import Counter
top_n = 30
class_counts = Counter(y_true_labels)
top_classes = [cls for cls, _ in class_counts.most_common(top_n)]

# Filtrar matriz de confusión y etiquetas
cm_top = confusion_matrix(
    [y if y in top_classes else 'Other' for y in y_true_labels],
    [y if y in top_classes else 'Other' for y in y_pred_labels],
    labels=top_classes + ['Other']
)

plt.figure(figsize=(24, 18))  # Mucho más grande
sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues', xticklabels=top_classes + ['Other'], yticklabels=top_classes + ['Other'])
plt.title('Confusion Matrix (Top 30 Classes + Other)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix plot saved to confusion_matrix.png (top 30 classes + Other)")
