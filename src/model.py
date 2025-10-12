import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
import pandas as pd
from PIL import Image
import os
from pathlib import Path

# Define the custom dataset
class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, image_processor, image_dir, unit_vocab=None):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir

        # Handle 'value' column, fill NaNs and convert to float
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce').fillna(0)

        # Build or use unit vocabulary
        if unit_vocab:
            self.unit_vocab = unit_vocab
        else:
            unique_units = self.df['unit'].astype(str).unique().tolist()
            self.unit_vocab = {unit: i+1 for i, unit in enumerate(unique_units)} 
            self.unit_vocab['__unknown__'] = 0
        
        self.unit_to_idx = self.unit_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Text data
        item_name = str(row.get('item_name', ''))
        description = str(row.get('description', ''))
        text = item_name + " " + description
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        
        # 2. Image data from local storage
        try:
            image_filename = Path(row['image_link']).name
            image_path = os.path.join(self.image_dir, image_filename)
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.image_processor(image, return_tensors='pt').pixel_values
        except Exception as e:
            # print(f"Warning: Could not load image {image_path}. Using a zero tensor. Error: {e}")
            pixel_values = torch.zeros((1, 3, 224, 224))

        # 3. Tabular data
        value = torch.tensor(row['value'], dtype=torch.float32)
        unit = str(row.get('unit', '__unknown__'))
        unit_idx = torch.tensor(self.unit_to_idx.get(unit, 0), dtype=torch.long)

        # 4. Price (target) - Apply log transformation
        price = torch.tensor(np.log1p(row['price']), dtype=torch.float32) if 'price' in self.df.columns else torch.tensor(0, dtype=torch.float32)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values.squeeze(0),
            'value': value,
            'unit_idx': unit_idx,
            'price': price
        }

# Gated Multimodal Unit
class GatedMultimodalUnit(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim_1, output_dim)
        self.linear_2 = nn.Linear(input_dim_2, output_dim)
        self.linear_gate_1 = nn.Linear(input_dim_1, output_dim)
        self.linear_gate_2 = nn.Linear(input_dim_2, output_dim)

    def forward(self, x1, x2):
        h1 = torch.tanh(self.linear_1(x1))
        h2 = torch.tanh(self.linear_2(x2))
        z = torch.sigmoid(self.linear_gate_1(x1) + self.linear_gate_2(x2))
        return z * h1 + (1 - z) * h2

# Define the Final Tri-Modal Model
class CrossModalAttentionModel(nn.Module):
    def __init__(self, unit_vocab_size, unit_embedding_dim=16, text_model_name='bert-base-uncased', image_model_name='google/vit-base-patch16-224-in21k'):
        super().__init__()
        
        # 1. Text encoder
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        # 2. Image encoder
        self.image_encoder = ViTModel.from_pretrained(image_model_name)
        image_hidden_size = self.image_encoder.config.hidden_size

        if text_hidden_size != image_hidden_size:
            self.image_proj = nn.Linear(image_hidden_size, text_hidden_size)
        else:
            self.image_proj = nn.Identity()

        # 3. Tabular tower
        self.unit_embedding = nn.Embedding(unit_vocab_size, unit_embedding_dim)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(1 + unit_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # 4. Cross-attention layer for Text-Image fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=8, batch_first=True)
        
        # Projection for Text-Image features to match GMU input
        self.text_image_proj = nn.Linear(text_hidden_size * 2, 128)

        # 5. Gated Multimodal Unit for final fusion
        self.gmu = GatedMultimodalUnit(input_dim_1=128, input_dim_2=128, output_dim=128)

        # 6. Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values, value, unit_idx):
        # 1. Get text embeddings
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state

        # 2. Get image embeddings
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_embeds = self.image_proj(image_outputs.last_hidden_state)
        
        # 3. Get tabular embeddings
        unit_embeds = self.unit_embedding(unit_idx)
        value_unsqueezed = value.unsqueeze(1)
        tabular_inputs = torch.cat((value_unsqueezed, unit_embeds), dim=1)
        tabular_embeds = self.tabular_mlp(tabular_inputs)

        # 4. Apply cross-attention for text-image
        attn_output, _ = self.cross_attention(query=text_embeds, key=image_embeds, value=image_embeds)
        
        # 5. Fuse features
        fused_text_image = torch.cat((text_embeds[:, 0, :], attn_output[:, 0, :]), dim=1)
        projected_text_image = self.text_image_proj(fused_text_image)

        # Use GMU to fuse text/image with tabular features
        final_features = self.gmu(projected_text_image, tabular_embeds)
        
        # 6. Predict the price
        price = self.regression_head(final_features)
        
        return price

if __name__ == '__main__':
    # This block is for testing the model's forward pass with dummy data
    print("Testing model architecture...")
    
    # Dummy data setup
    batch_size = 4
    dummy_df = pd.DataFrame({
        'item_name': ['Product A', 'Product B', 'Product C', 'Product D'],
        'description': ['Desc A', 'Desc B', 'Desc C', 'Desc D'],
        'image_link': ['', '', '', ''], # No images downloaded in test
        'value': [10, 20, 30, 40],
        'unit': ['kg', 'lb', 'kg', 'oz'],
        'price': [1.0, 2.0, 3.0, 4.0]
    })

    # Initializations
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Create dataset and get vocab size
    dataset = ProductDataset(dataframe=dummy_df, tokenizer=tokenizer, image_processor=image_processor)
    unit_vocab_size = len(dataset.unit_vocab)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Initialize the model
    model = CrossModalAttentionModel(unit_vocab_size=unit_vocab_size)
    print("Model initialized successfully.")

    # Process a single batch to test the forward pass
    try:
        batch = next(iter(dataloader))
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            value=batch['value'],
            unit_idx=batch['unit_idx']
        )
        print(f"Model output shape: {outputs.shape}")
        print(f"Sample predictions: {outputs.squeeze().tolist()}")
        print("\nModel forward pass test successful!")
    except Exception as e:
        print(f"\nError during model forward pass test: {e}")
