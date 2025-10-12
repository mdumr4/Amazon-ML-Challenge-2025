import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, ViTImageProcessor
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Import our model and dataset classes from model.py
from model import ProductDataset, CrossModalAttentionModel

# --- Custom Loss Function ---
class SmoothSMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Ensure predictions are positive as per problem constraints
        y_pred = torch.relu(y_pred)
        
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        
        # Add epsilon to the denominator for numerical stability
        loss = torch.mean(numerator / (denominator + self.epsilon))
        
        return loss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV_PATH = "dataset/train_cleaned.csv"
IMAGE_DIR = "dataset/images"
MODEL_SAVE_PATH = "cross_modal_model.pth"
EPOCHS = 10
BATCH_SIZE = 16 # Adjust based on your GPU memory
LEARNING_RATE = 5e-6 # Low learning rate for fine-tuning

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        value = batch['value'].to(device)
        unit_idx = batch['unit_idx'].to(device)
        prices = batch['price'].to(device)

        # Forward pass
        predictions = model(input_ids, attention_mask, pixel_values, value, unit_idx)

        # Calculate loss
        loss = loss_fn(predictions.squeeze(), prices)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            value = batch['value'].to(device)
            unit_idx = batch['unit_idx'].to(device)
            prices = batch['price'].to(device)

            # Forward pass
            predictions = model(input_ids, attention_mask, pixel_values, value, unit_idx)

            # Calculate loss
            loss = loss_fn(predictions.squeeze(), prices)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Load and Split Data ---
    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # --- 2. Initialize Tokenizer and Image Processor ---
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # --- 3. Create Datasets and DataLoaders ---
    print("Creating datasets...")
    # Create train dataset and its unit vocabulary
    train_dataset = ProductDataset(dataframe=train_df, tokenizer=tokenizer, image_processor=image_processor, image_dir=IMAGE_DIR)
    unit_vocab = train_dataset.unit_vocab
    
    # Share the vocabulary with the validation dataset
    val_dataset = ProductDataset(dataframe=val_df, tokenizer=tokenizer, image_processor=image_processor, image_dir=IMAGE_DIR, unit_vocab=unit_vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print("Datasets created successfully.")

    # --- 4. Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    unit_vocab_size = len(unit_vocab)
    model = CrossModalAttentionModel(unit_vocab_size=unit_vocab_size).to(DEVICE)
    
    loss_fn = SmoothSMAPELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized successfully.")

    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, DEVICE)
        val_loss = validate_one_epoch(model, val_dataloader, loss_fn, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # --- 6. Save the Model ---
    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()