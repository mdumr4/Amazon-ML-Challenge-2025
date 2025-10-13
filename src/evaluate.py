import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoImageProcessor
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from collections import OrderedDict

# Import model and dataset classes
from model import ProductDataset, CrossModalAttentionModel

# --- Metric Calculation ---
def calculate_metrics(y_true, y_pred):
    y_pred[y_pred < 0] = 0.01 # Safeguard
    
    # SMAPE
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {'SMAPE': smape, 'MAE': mae, 'RMSE': rmse}

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Absolute paths for Google Drive environment
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "dataset/train_features.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "dataset/images")
MODEL_PATH = os.path.join(BASE_PATH, "best.pth")
VOCAB_PATH = os.path.join(BASE_PATH, "unit_vocab.json")
STATS_PATH = os.path.join(BASE_PATH, "tabular_stats.json")
BATCH_SIZE = 128

def main():
    cudnn.benchmark = True
    print("Starting evaluation process...")
    print(f"Using device: {DEVICE}")

    # --- 1. Load Vocab, Tokenizer, and Image Processor ---
    print("Loading vocabulary and processors...")
    with open(VOCAB_PATH, 'r') as f:
        unit_vocab = json.load(f)
    unit_vocab_size = len(unit_vocab)
    with open(STATS_PATH, 'r') as f:
        tabular_stats = json.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # --- 2. Load Model ---
    print(f"Loading trained model from {MODEL_PATH}...")
    model = CrossModalAttentionModel(
        unit_vocab_size=unit_vocab_size, 
        num_numerical_features=len(tabular_stats['mean'])
    ).to(DEVICE)
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 3. Create Validation Dataset and DataLoader ---
    print("Loading and preparing validation data...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)

    val_dataset = ProductDataset(
        dataframe=val_df, 
        tokenizer=tokenizer, 
        image_dir=IMAGE_DIR, 
        unit_vocab=unit_vocab,
        tabular_stats=tabular_stats,
        split='val'
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # --- 4. Generate Predictions ---
    print("Generating predictions on the validation set...")
    all_log_preds = []
    all_log_prices = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            pixel_values = batch['pixel_values'].to(DEVICE, non_blocking=True)
            tabular_features = batch['tabular_features'].to(DEVICE, non_blocking=True)
            unit_idx = batch['unit_idx'].to(DEVICE, non_blocking=True)
            prices = batch['price'].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=str(DEVICE)):
                log_preds = model(input_ids, attention_mask, pixel_values, tabular_features, unit_idx)
            
            all_log_preds.append(log_preds.cpu())
            all_log_prices.append(prices.cpu())

    # --- 5. Calculate Metrics and Analyze Errors ---
    print("Calculating final metrics and analyzing errors...")
    log_preds_cat = torch.cat(all_log_preds).numpy()
    log_prices_cat = torch.cat(all_log_prices).numpy()

    final_preds = np.expm1(log_preds_cat.flatten())
    final_prices = np.expm1(log_prices_cat.flatten())

    metrics = calculate_metrics(final_prices, final_preds)
    print("\n--- Final Validation Metrics ---")
    print(f"  SMAPE: {metrics['SMAPE']:.4f}%")
    print(f"  MAE:   ${metrics['MAE']:.2f}")
    print(f"  RMSE:  ${metrics['RMSE']:.2f}")
    print("---------------------------------")

    # --- Error Analysis: Find Worst Offenders ---
    error_df = val_df.copy()
    error_df['predicted_price'] = final_preds
    error_df['actual_price'] = final_prices
    error_df['absolute_error'] = np.abs(error_df['predicted_price'] - error_df['actual_price'])
    error_df['percentage_error'] = (error_df['absolute_error'] / error_df['actual_price']) * 100

    worst_predictions = error_df.sort_values(by='absolute_error', ascending=False)

    print("\n--- Top 50 Worst Predictions (by Absolute Error) ---")
    print(worst_predictions[['sample_id', 'item_name', 'actual_price', 'predicted_price', 'absolute_error', 'percentage_error']].head(50))
    print("-----------------------------------------------------")

    # --- Plotting ---
    plt.figure(figsize=(10, 10))
    plt.scatter(final_prices, final_preds, alpha=0.3)
    plt.plot([min(final_prices), max(final_prices)], [min(final_prices), max(final_prices)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plot_filename = 'prediction_vs_actual.png'
    plt.savefig(plot_filename)
    print(f"\nScatter plot saved to {plot_filename}")

if __name__ == '__main__':
    main()