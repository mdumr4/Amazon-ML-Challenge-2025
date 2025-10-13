import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, ViTImageProcessor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from tqdm import tqdm

# Import our model and dataset classes from model.py
from model import ProductDataset, CrossModalAttentionModel

# --- Custom Loss & Metric Functions ---
class SmoothSMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.relu(y_pred)
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        loss = torch.mean(numerator / (denominator + self.epsilon))
        return loss

def calculate_metrics(y_true, y_pred):
    # Ensure predictions are positive
    y_pred[y_pred < 0] = 0
    
    # SMAPE
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100 # Add epsilon to avoid division by zero

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    return {'SMAPE': smape, 'MAE': mae, 'RMSE': rmse}

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV_PATH = "dataset/train_cleaned.csv"
IMAGE_DIR = "dataset/images"
LATEST_MODEL_SAVE_PATH = "latest.pth"
BEST_MODEL_SAVE_PATH = "best.pth"
VOCAB_SAVE_PATH = "unit_vocab.json"
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 5e-6

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad(set_to_none=True)
        
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        value = batch['value'].to(device, non_blocking=True)
        unit_idx = batch['unit_idx'].to(device, non_blocking=True)
        prices = batch['price'].to(device, non_blocking=True)

        with autocast():
            predictions = model(input_ids, attention_mask, pixel_values, value, unit_idx)
            loss = loss_fn(predictions.squeeze(), prices)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_log_preds = []
    all_log_prices = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            value = batch['value'].to(device, non_blocking=True)
            unit_idx = batch['unit_idx'].to(device, non_blocking=True)
            prices = batch['price'].to(device, non_blocking=True)

            with autocast():
                log_preds = model(input_ids, attention_mask, pixel_values, value, unit_idx)
                loss = loss_fn(log_preds.squeeze(), prices)
            
            total_loss += loss.item()
            all_log_preds.append(log_preds.cpu())
            all_log_prices.append(prices.cpu())

    # Calculate final metrics
    log_preds_cat = torch.cat(all_log_preds).numpy()
    log_prices_cat = torch.cat(all_log_prices).numpy()

    # Inverse transform to get actual prices
    final_preds = np.expm1(log_preds_cat)
    final_prices = np.expm1(log_prices_cat)

    metrics = calculate_metrics(final_prices, final_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics

def main():
    cudnn.benchmark = True
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    print("Creating datasets...")
    train_dataset = ProductDataset(dataframe=train_df, tokenizer=tokenizer, image_processor=image_processor, image_dir=IMAGE_DIR)
    unit_vocab = train_dataset.unit_vocab
    val_dataset = ProductDataset(dataframe=val_df, tokenizer=tokenizer, image_processor=image_processor, image_dir=IMAGE_DIR, unit_vocab=unit_vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    print("Datasets created successfully.")

    print("Initializing model...")
    unit_vocab_size = len(unit_vocab)
    model = CrossModalAttentionModel(unit_vocab_size=unit_vocab_size).to(DEVICE)
    
    # Disabled torch.compile due to instability
    # model = torch.compile(model, mode="max-autotune")
    
    loss_fn = SmoothSMAPELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    print("Model initialized successfully.")

    best_val_smape = float('inf')

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, DEVICE, epoch, scaler)
        val_loss, metrics = validate_one_epoch(model, val_dataloader, loss_fn, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SMAPE: {metrics['SMAPE']:.4f}% | Val MAE: ${metrics['MAE']:.2f}")

        # Save latest model
        torch.save(model.state_dict(), LATEST_MODEL_SAVE_PATH)

        # Save best model based on SMAPE
        if metrics['SMAPE'] < best_val_smape:
            best_val_smape = metrics['SMAPE']
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"  -> New best model saved with SMAPE: {best_val_smape:.4f}%")

    print(f"Training complete. Best validation SMAPE: {best_val_smape:.4f}%")
    print(f"Saving unit vocabulary to {VOCAB_SAVE_PATH}...")
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(unit_vocab, f)
    print("Vocabulary saved successfully.")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()