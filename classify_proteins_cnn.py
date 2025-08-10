import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Added confusion_matrix

import matplotlib.pyplot as plt # New import
import seaborn as sns # New import

# --- Part 3: Protein Classification with CNN ---

print("\n--- Starting Protein Classification with CNN ---")

# Load the processed data (assuming it's saved from previous steps)
try:
    df_processed = pd.read_csv("cleaned_and_embedded_protein_data.tsv", sep='\t')
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
        'Pfam_Family_ID': ['PF00001', 'PF00002', 'PF00001', 'PF00003', 'Unknown', 'PF00001', 'PF00002', 'PF00001', 'PF00003', 'Unknown'],
        'Protein class': ['Enzyme', 'Transporter', 'Enzyme', 'Structural', 'Enzyme', 'Transporter', 'Enzyme', 'Structural', 'Enzyme', 'Transporter'],
        'Biological process': ['Metabolism', 'Transport', 'Signaling', 'Structure', 'Metabolism', 'Transport', 'Signaling', 'Structure', 'Metabolism', 'Transport'],
        'Molecular function': ['Catalytic', 'Binding', 'Catalytic', 'Structural', 'Binding', 'Catalytic', 'Binding', 'Catalytic', 'Structural', 'Binding'],
        'ESM2_Embedding': [np.random.rand(320) for _ in range(10)] # Dummy embeddings (ESM2_t12_35M_UR50D has 320 dim)
    }
    df_processed = pd.DataFrame(data)
    print("Dummy DataFrame created.")

# Prepare Features (X) and Target (y)
# Ensure ESM2_Embedding is in a format suitable for PyTorch (float32)
# Stack embeddings and convert to PyTorch tensor
X = np.vstack(df_processed['ESM2_Embedding'].values).astype(np.float32)
X = torch.tensor(X).unsqueeze(1) # Add a channel dimension for Conv1d (batch_size, channels, sequence_length)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_processed['Protein class'])
y = torch.tensor(y, dtype=torch.long) # Convert to long for CrossEntropyLoss

num_classes = len(label_encoder.classes_)
embedding_dim = X.shape[2] # Dimension of the ESM2 embedding

print(f"Shape of features (X) for CNN: {X.shape} (batch_size, channels, embedding_dim)")
print(f"Shape of target (y): {y.shape}")
print(f"Number of classes: {num_classes}")
print(f"Embedding dimension: {embedding_dim}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create PyTorch Datasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32 # You can adjust this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN Model
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ProteinCNN, self).__init__()
        # Conv1d layer: in_channels=1 (for single embedding), out_channels=128, kernel_size=3
        # We'll use a kernel size that makes sense for the embedding dimension.
        # A smaller kernel size will look for local patterns within the embedding.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the output size after conv and pool to determine input for linear layer
        # (embedding_dim + 2*padding - kernel_size) / stride + 1  (for conv)
        # (output_conv_dim - kernel_size) / stride + 1 (for pool)
        # Here, with padding=1, kernel_size=3, stride=1 for conv: output_conv_dim = embedding_dim
        # With pool kernel_size=2, stride=2: output_pool_dim = embedding_dim // 2
        
        self.fc1 = nn.Linear(128 * (embedding_dim // 2), 64) # Adjust input features based on pooling output
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten the output for the fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ProteinCNN(embedding_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10 # You can adjust this
print(f"\nTraining CNN for {num_epochs} epochs...")
train_losses = [] # New list to store training losses

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss) # Store the loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("\nCNN Training Complete.")

# Save the trained model
model_save_path = "protein_cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

# Evaluate the Model
print("\nEvaluating the CNN model...")
model.eval() # Set model to evaluation mode
y_true = []
y_pred = []

with torch.no_grad(): # Disable gradient calculation during evaluation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1) # Get the class with the highest probability
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Convert numerical predictions back to original class names for report
y_true_labels = label_encoder.inverse_transform(y_true)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print(f"Accuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

print("\n--- Protein Classification with CNN Complete ---")

# Plotting Training Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png') # Save the plot
plt.close() # Close the plot to free memory
print("\nTraining loss plot saved to training_loss.png")

# Plotting Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig('confusion_matrix.png') # Save the plot
plt.close() # Close the plot to free memory
print("Confusion matrix plot saved to confusion_matrix.png")