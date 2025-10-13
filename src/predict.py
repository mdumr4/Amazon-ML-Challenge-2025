import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
from tqdm import tqdm
import os
import torch.backends.cudnn as cudnn

# Import model and dataset classes
from model import ProductDataset, CrossModalAttentionModel

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV_PATH = "dataset/test_cleaned.csv"
IMAGE_DIR = "dataset/images"
MODEL_PATH = "cross_modal_model.pth"
VOCAB_PATH = "unit_vocab.json"
OUTPUT_CSV_PATH = "test_out.csv"
BATCH_SIZE = 256 # Can use a larger batch size for inference

def main():
    # --- Performance Optimizations ---
    cudnn.benchmark = True

    print("Starting prediction process...")
    print(f"Using device: {DEVICE}")

    # --- 1. Load Vocab, Tokenizer, and Image Processor ---
    print("Loading vocabulary and processors...")
    with open(VOCAB_PATH, 'r') as f:
        unit_vocab = json.load(f)
    unit_vocab_size = len(unit_vocab)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # --- 2. Load Model ---
    print("Loading trained model...")
    model = CrossModalAttentionModel(unit_vocab_size=unit_vocab_size).to(DEVICE)
    
    # Compile the model before loading weights for max performance
    print("Compiling model...")
    model = torch.compile(model)

    # Load the state dict from the saved file (it will have _orig_mod prefixes)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval() # Set model to evaluation mode

    # --- 3. Create Test Dataset and DataLoader ---
    print("Loading and preparing test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_dataset = ProductDataset(
        dataframe=test_df, 
        tokenizer=tokenizer, 
        image_processor=image_processor, 
        image_dir=IMAGE_DIR, 
        unit_vocab=unit_vocab
    )
    # Use num_workers for faster data loading, even in prediction
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # --- 4. Generate Predictions ---
    print("Generating predictions...")
    predictions_list = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            pixel_values = batch['pixel_values'].to(DEVICE, non_blocking=True)
            value = batch['value'].to(DEVICE, non_blocking=True)
            unit_idx = batch['unit_idx'].to(DEVICE, non_blocking=True)

            # Use autocast for mixed precision inference
            with torch.autocast(device_type=str(device)):
                log_preds = model(input_ids, attention_mask, pixel_values, value, unit_idx)
            
            # Inverse transform and safeguard
            price_preds = np.expm1(log_preds.cpu().numpy())
            price_preds[price_preds < 0] = 0.01 # Safeguard in a vectorized way
            predictions_list.extend(price_preds.flatten().tolist())

    # --- 5. Create Submission File ---
    print("Creating submission file...")
    # Ensure the number of predictions matches the dataframe length
    if len(predictions_list) != len(test_df):
        raise ValueError(f"Mismatch in prediction count! Expected {len(test_df)}, got {len(predictions_list)}")

    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions_list
    })

    submission_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Submission file saved to {OUTPUT_CSV_PATH}")
    print(f"Total predictions: {len(submission_df)}")
    print("Sample predictions:")
    print(submission_df.head())

if __name__ == '__main__':
    main()
