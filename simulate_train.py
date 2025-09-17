import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from collections import Counter
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TherapySequenceDataset(Dataset):
    """Dataset class for therapy module sequences"""
    
    def __init__(self, sequences, effectiveness_scores, module_to_idx):
        self.sequences = sequences
        self.effectiveness_scores = effectiveness_scores
        self.module_to_idx = module_to_idx
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        effectiveness = self.effectiveness_scores[idx]
        
        # Convert modules to indices
        sequence_indices = [self.module_to_idx[module] for module in sequence]
        
        return torch.tensor(sequence_indices, dtype=torch.long), torch.tensor(effectiveness, dtype=torch.float32)

class TherapySequenceModel(nn.Module):
    """LSTM-based model for predicting therapy effectiveness from module sequences"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(TherapySequenceModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Pack padded sequences for efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, _) = self.lstm(packed)
        
        # Use the last hidden state for prediction
        # hidden shape: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]  # Take the last layer
        
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        
        return output.squeeze()

def generate_synthetic_data(n_sequences=500):
    """Generate synthetic therapy module sequences with effectiveness scores"""
    
    modules = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    
    # Define some "effective" patterns (for simulation purposes)
    effective_patterns = [
        ['A', 'B', 'A'],  # Pattern 1
        ['C', 'D', 'C'],  # Pattern 2
        ['A', 'F', 'F'],  # Pattern 3
        ['B', 'C', 'B'],  # Pattern 4
    ]
    
    sequences = []
    effectiveness_scores = []
    
    for _ in range(n_sequences):
        # Generate random sequence length between 4 and 20
        seq_length = np.random.randint(4, 21)
        
        # Generate base effectiveness (random component)
        base_effectiveness = np.random.normal(0.5, 0.1)
        
        # Generate sequence
        if np.random.random() < 0.3:  # 30% chance of including effective pattern
            # Start with an effective pattern
            pattern = random.choice(effective_patterns)
            sequence = pattern.copy()
            base_effectiveness += 0.2  # Boost for effective pattern
        else:
            sequence = []
        
        # Fill rest of sequence with random modules
        while len(sequence) < seq_length:
            sequence.append(np.random.choice(modules))
        
        # Add effectiveness bonus for certain modules
        module_bonuses = {'A': 0.05, 'F': 0.03, 'C': 0.02}
        for module in sequence:
            if module in module_bonuses:
                base_effectiveness += module_bonuses[module]
        
        # Add penalty for very long sequences (therapy fatigue)
        if len(sequence) > 15:
            base_effectiveness -= 0.1
        
        # Clip effectiveness between 0 and 1
        base_effectiveness = np.clip(base_effectiveness, 0, 1)
        
        sequences.append(sequence)
        effectiveness_scores.append(base_effectiveness)
    
    return sequences, effectiveness_scores

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    sequences, effectiveness_scores = zip(*batch)
    
    # Get sequence lengths
    lengths = [len(seq) for seq in sequences]
    
    # Pad sequences to the same length
    max_length = max(lengths)
    padded_sequences = []
    
    for seq in sequences:
        padded = list(seq) + [0] * (max_length - len(seq))  # Pad with 0 (will be ignored)
        padded_sequences.append(padded)
    
    return (torch.tensor(padded_sequences, dtype=torch.long), 
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(effectiveness_scores, dtype=torch.float32))

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """Train the therapy sequence model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for sequences, lengths, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths, targets in val_loader:
                outputs = model(sequences, lengths)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 20:  # Early stopping patience
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def predict_subsequence_effectiveness(model, subsequence, module_to_idx):
    """Predict effectiveness for a given subsequence"""
    model.eval()
    
    # Convert subsequence to indices
    sequence_indices = [module_to_idx[module] for module in subsequence]
    sequence_tensor = torch.tensor([sequence_indices], dtype=torch.long)
    lengths = torch.tensor([len(sequence_indices)], dtype=torch.long)
    
    with torch.no_grad():
        prediction = model(sequence_tensor, lengths)
    
    return prediction.item()

# Main execution
if __name__ == "__main__":
    print("Generating synthetic therapy sequence data...")
    sequences, effectiveness_scores = generate_synthetic_data(n_sequences=500)
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Average sequence length: {np.mean([len(seq) for seq in sequences]):.2f}")
    print(f"Average effectiveness: {np.mean(effectiveness_scores):.3f}")
    
    # Create module vocabulary
    all_modules = set()
    for seq in sequences:
        all_modules.update(seq)
    
    module_to_idx = {module: idx+1 for idx, module in enumerate(sorted(all_modules))}  # Start from 1, 0 is padding
    module_to_idx['<PAD>'] = 0
    idx_to_module = {idx: module for module, idx in module_to_idx.items()}
    
    print(f"Vocabulary size: {len(module_to_idx)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, effectiveness_scores, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = TherapySequenceDataset(X_train, y_train, module_to_idx)
    val_dataset = TherapySequenceDataset(X_val, y_val, module_to_idx)
    test_dataset = TherapySequenceDataset(X_test, y_test, module_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Initialize and train model
    vocab_size = len(module_to_idx)
    embedding_dim = 32
    hidden_dim = 64
    
    model = TherapySequenceModel(vocab_size, embedding_dim, hidden_dim)
    
    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=100)
    
    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for sequences, lengths, targets in test_loader:
            outputs = model(sequences, lengths)
            test_predictions.extend(outputs.numpy())
            test_targets.extend(targets.numpy())
    
    mse = mean_squared_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # Example predictions
    print("\nExample subsequence predictions:")
    test_sequences = [
        ['A', 'B', 'A'],
        ['C', 'D', 'C'],
        ['A', 'F', 'F'],
        ['B', 'C', 'B'],
        ['Q', 'P', 'O', 'N']
    ]
    
    for seq in test_sequences:
        pred = predict_subsequence_effectiveness(model, seq, module_to_idx)
        print(f"Sequence {' → '.join(seq)}: Predicted effectiveness = {pred:.3f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_targets, test_predictions, alpha=0.6)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--', lw=2)
    plt.xlabel('Actual Effectiveness')
    plt.ylabel('Predicted Effectiveness')
    plt.title(f'Predictions vs Actual (R² = {r2:.3f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')  # Save instead of show
    plt.show(block=False)  # Don't pause
    plt.pause(5)           

    
    print("\nModel training complete!")
    print("The model can now predict effectiveness for new therapy module sequences.")


    # Extract the Embeddings
    # After training your model
    embedding_weights = model.embedding.weight.data.numpy()  # Shape: (17, embedding_dim)

    # Get embedding for module 'A'
    module_A_embedding = embedding_weights[module_to_idx['A']]  # Shape: (embedding_dim,)



    # 1. 2D Plot with t-SNE/UMAP
    # Reduce to 2D for plotting
    # For 17 modules, use smaller perplexity
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # perplexity < 17
    embeddings_2d = tsne.fit_transform(embedding_weights[1:])  # Skip padding token

    # Plot
    plt.figure(figsize=(10, 8))
    for i, module in enumerate(sorted(all_modules)):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100)
        plt.annotate(module, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.title("Therapy Module Embeddings (t-SNE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('embeddings_2d.png', dpi=300, bbox_inches='tight')  # Save instead of show
    plt.show(block=False)  # Don't pause
    plt.pause(5)      

    from sklearn.metrics.pairwise import cosine_similarity

    # Find most similar modules to 'A'
    similarities = cosine_similarity([embedding_weights[module_to_idx['A']]], 
                                embedding_weights[1:])[0]

    # Get top similar modules
    similar_modules = sorted(zip(sorted(all_modules), similarities), 
                            key=lambda x: x[1], reverse=True)
    print("Modules most similar to A:", similar_modules[:5])
    # The visualization will show you which therapy modules the model considers "similar" based on their usage patterns in your sequences
