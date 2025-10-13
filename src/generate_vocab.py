import pandas as pd
import json
from transformers import BertTokenizer, ViTImageProcessor

# Import our model and dataset classes from model.py
# We only need ProductDataset to generate the vocab
from model import ProductDataset

# --- Configuration ---
TRAIN_CSV_PATH = "dataset/train_cleaned.csv"
IMAGE_DIR = "dataset/images" # This is needed for the dataset constructor, but images won't be loaded
VOCAB_SAVE_PATH = "unit_vocab.json"

def main():
    print("Generating unit vocabulary from training data...")

    # Load the training dataframe
    df = pd.read_csv(TRAIN_CSV_PATH)
    
    # We need to initialize tokenizer and image_processor for the dataset constructor
    # but they won't be used to generate the vocab.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Create a dataset instance. This will automatically build the vocabulary.
    train_dataset = ProductDataset(
        dataframe=df, 
        tokenizer=tokenizer, 
        image_processor=image_processor, 
        image_dir=IMAGE_DIR
    )
    unit_vocab = train_dataset.unit_vocab

    # Save the vocabulary to a file
    print(f"Saving unit vocabulary to {VOCAB_SAVE_PATH}...")
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(unit_vocab, f, indent=4)
    print("Vocabulary saved successfully.")
    print(f"Found {len(unit_vocab)} unique units.")

if __name__ == '__main__':
    main()
