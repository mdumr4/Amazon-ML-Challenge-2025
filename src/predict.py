import torch
import pandas as pd
import numpy as np
import json
import joblib
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoImageProcessor
from tqdm import tqdm
import os
import torch.backends.cudnn as cudnn
from collections import OrderedDict

# Import model and dataset classes
from model import ProductDataset, CrossModalAttentionModel

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"
TEST_CSV_PATH = os.path.join(BASE_PATH, "dataset/test_features.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "dataset/images")
NN_MODEL_PATH = os.path.join(BASE_PATH, "best.pth") # Neural Network model
LGBM_MODEL_PATH = os.path.join(BASE_PATH, "lightgbm_model.pkl") # LightGBM model
VOCAB_PATH = os.path.join(BASE_PATH, "unit_vocab.json")
STATS_PATH = os.path.join(BASE_PATH, "tabular_stats.json")
UNIT_ENCODER_PATH = os.path.join(BASE_PATH, "unit_label_encoder.pkl") # From LGBM training
OUTPUT_CSV_PATH = os.path.join(BASE_PATH, "test_out.csv")
BATCH_SIZE = 256

def main():
    cudnn.benchmark = True
    print("Starting ENSEMBLE prediction process...")
    print(f"Using device: {DEVICE}")

    # --- 1. Load All Models and Supporting Files ---
    print("Loading vocabularies, encoders, and processors...")
    with open(VOCAB_PATH, 'r') as f:
        unit_vocab = json.load(f)
    unit_vocab_size = len(unit_vocab)
    
    with open(STATS_PATH, 'r') as f:
        tabular_stats = json.load(f)

    unit_encoder = joblib.load(UNIT_ENCODER_PATH)
    lgbm_model = joblib.load(LGBM_MODEL_PATH)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Loading Neural Network model...")
    nn_model = CrossModalAttentionModel(
        unit_vocab_size=unit_vocab_size, 
        num_numerical_features=2
    ).to(DEVICE)
    state_dict = torch.load(NN_MODEL_PATH, map_location=DEVICE)
    nn_model.load_state_dict(state_dict)
    nn_model.eval()

    # --- 2. Prepare Data ---
    print("Loading and preparing test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # --- 3. Generate Predictions from LightGBM Model ---
    print("Generating predictions from LightGBM model...")
    lgbm_test_df = test_df.copy()
    lgbm_test_df['quantity_unit'] = unit_encoder.transform(lgbm_test_df['quantity_unit'].astype(str))
    lgbm_features = lgbm_test_df[['pack_size', 'quantity_value', 'quantity_unit']]
    log_preds_lgbm = lgbm_model.predict(lgbm_features)

    # --- 4. Generate Predictions from Neural Network Model ---
    print("Generating predictions from Neural Network model...")
    test_dataset = ProductDataset(
        dataframe=test_df, 
        tokenizer=tokenizer, 
        image_dir=IMAGE_DIR, 
        unit_vocab=unit_vocab,
        tabular_stats=tabular_stats,
        split='test'
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    log_preds_nn_list = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting with NN"):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            pixel_values = batch['pixel_values'].to(DEVICE, non_blocking=True)
            tabular_features = batch['tabular_features'].to(DEVICE, non_blocking=True)
            unit_idx = batch['unit_idx'].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=str(DEVICE)):
                log_preds = nn_model(input_ids, attention_mask, pixel_values, tabular_features, unit_idx)
            log_preds_nn_list.append(log_preds.cpu())
    
    log_preds_nn = torch.cat(log_preds_nn_list).numpy()

    # --- 5. Ensemble Predictions and Post-Process ---
    print("Ensembling predictions...")
    # Simple average of the log-space predictions
    final_log_preds = (log_preds_lgbm + log_preds_nn.flatten()) / 2.0

    # Inverse transform and safeguard
    final_prices = np.expm1(final_log_preds)
    final_prices[final_prices < 0] = 0.01

    # --- 6. Create Submission File ---
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_prices
    })

    submission_df.to_csv(OUTPUT_CSV_PATH, index=False, header=False)
    print(f"Submission file saved to {OUTPUT_CSV_PATH}")
    print("Sample predictions:")
    print(submission_df.head())

if __name__ == '__main__':
    main()