import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- Configuration ---
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "dataset/train_features.csv")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "lightgbm_model.pkl")
ENCODER_SAVE_PATH = os.path.join(BASE_PATH, "unit_label_encoder.pkl")

def main():
    print("Starting LightGBM model training...")

    # --- 1. Load Data ---
    df = pd.read_csv(TRAIN_CSV_PATH)
    print("Data loaded successfully.")

    # --- 2. Feature Engineering & Preprocessing ---
    # Log transform the target variable
    df['price'] = np.log1p(df['price'])

    # Select features
    features = ['pack_size', 'quantity_value', 'quantity_unit']
    target = 'price'

    # Label encode the categorical feature
    unit_encoder = LabelEncoder()
    df['quantity_unit'] = unit_encoder.fit_transform(df['quantity_unit'].astype(str))
    print("Categorical features encoded.")

    # --- 3. Train/Validation Split ---
    X = df[features]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Model Training ---
    print("Training LightGBM model...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', # MAE Loss, robust to outliers
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )

    # --- 5. Save Model and Encoder ---
    print("Saving model and encoder...")
    joblib.dump(lgbm, MODEL_SAVE_PATH)
    joblib.dump(unit_encoder, ENCODER_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Encoder saved to {ENCODER_SAVE_PATH}")

if __name__ == '__main__':
    main()
