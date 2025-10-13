import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, AutoImageProcessor, RobertaTokenizer
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
    y_pred[y_pred < 0] = 0.01
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'SMAPE': smape, 'MAE': mae, 'RMSE': rmse}

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "dataset/train_features.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "dataset/images")
LATEST_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "latest.pth")
BEST_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "best.pth")
VOCAB_SAVE_PATH = os.path.join(BASE_PATH, "unit_vocab.json")
STATS_SAVE_PATH = os.path.join(BASE_PATH, "tabular_stats.json")
EPOCHS = 10
BATCH_SIZE = 96
LEARNING_RATE = 5e-6
TEXT_MODEL_NAME = 'bert-base-uncased' # or 'roberta-large'

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        tabular_features = batch['tabular_features'].to(device, non_blocking=True)
        unit_idx = batch['unit_idx'].to(device, non_blocking=True)
        prices = batch['price'].to(device, non_blocking=True)

        with autocast():
            predictions = model(input_ids, attention_mask, pixel_values, tabular_features, unit_idx)
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
            tabular_features = batch['tabular_features'].to(device, non_blocking=True)
            unit_idx = batch['unit_idx'].to(device, non_blocking=True)
            prices = batch['price'].to(device, non_blocking=True)

            with autocast():
                log_preds = model(input_ids, attention_mask, pixel_values, tabular_features, unit_idx)
                loss = loss_fn(log_preds.squeeze(), prices)

            total_loss += loss.item()
            all_log_preds.append(log_preds.cpu())
            all_log_prices.append(prices.cpu())

    log_preds_cat = torch.cat(all_log_preds).numpy()
    log_prices_cat = torch.cat(all_log_prices).numpy()

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

    if 'roberta' in TEXT_MODEL_NAME:
        tokenizer = RobertaTokenizer.from_pretrained(TEXT_MODEL_NAME)
    else:
        tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    print("Creating datasets and calculating stats...")
    train_dataset = ProductDataset(dataframe=train_df, tokenizer=tokenizer, image_dir=IMAGE_DIR, split='train')
    unit_vocab = train_dataset.unit_vocab
    tabular_stats = train_dataset.tabular_stats

    val_dataset = ProductDataset(dataframe=val_df, tokenizer=tokenizer, image_dir=IMAGE_DIR, split='val', unit_vocab=unit_vocab, tabular_stats=tabular_stats)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    print("Datasets created successfully.")

    print("Initializing model...")
    model = CrossModalAttentionModel(
        unit_vocab_size=len(unit_vocab),
        num_numerical_features=len(train_dataset.numerical_cols),
        text_model_name=TEXT_MODEL_NAME,
        image_model_name='facebook/dinov2-base'
    ).to(DEVICE)

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

        torch.save(model.state_dict(), LATEST_MODEL_SAVE_PATH)

        if metrics['SMAPE'] < best_val_smape:
            best_val_smape = metrics['SMAPE']
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"  -> New best model saved with SMAPE: {best_val_smape:.4f}%")

    print(f"Training complete. Best validation SMAPE: {best_val_smape:.4f}%")

    print(f"Saving vocabularies to {VOCAB_SAVE_PATH} and {STATS_SAVE_PATH}...")
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(unit_vocab, f)
    # Convert stats to JSON serializable format
    stats_to_save = {
        'mean': tabular_stats['mean'].to_dict(),
        'std': tabular_stats['std'].to_dict()
    }
    with open(STATS_SAVE_PATH, 'w') as f:
        json.dump(stats_to_save, f)
    print("Vocabularies saved successfully.")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()