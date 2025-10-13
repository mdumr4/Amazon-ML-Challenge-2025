import pandas as pd
import os
from utils import download_images

# Define file paths
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "dataset/train_cleaned.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "dataset/test_cleaned.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "dataset/images")

def main():
    print("Starting image download process...")

    # Create the target directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created directory: {IMAGE_DIR}")

    # Load dataframes
    print("Loading CSV files...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Combine image links from both dataframes
    all_links = pd.concat([train_df['image_link'], test_df['image_link']], ignore_index=True)
    unique_links = all_links.unique().tolist()

    print(f"Found {len(unique_links)} unique image links to download.")

    # Download the images
    download_images(image_links=unique_links, download_folder=IMAGE_DIR)

    print("\nImage download process complete.")

if __name__ == '__main__':
    main()
