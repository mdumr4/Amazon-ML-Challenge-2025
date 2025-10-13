import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor, RobertaModel, RobertaTokenizer, AutoModel
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import re
from torchvision import transforms

# Define the new, advanced ProductDataset class
class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, image_dir, split='train', unit_vocab=None, tabular_stats=None):
        self.df = dataframe.copy()
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.split = split

        # --- Text Cleaning ---
        self.df['item_name'] = self.df['item_name'].astype(str).apply(self._clean_text)

        # --- Tabular Feature Processing ---
        self.numerical_cols = ['pack_size', 'quantity_value']
        for col in self.numerical_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        if self.split == 'train':
            self.tabular_stats = {
                'mean': self.df[self.numerical_cols].mean(),
                'std': self.df[self.numerical_cols].std()
            }
        else:
            if tabular_stats is None:
                raise ValueError("Must provide tabular_stats for val/test splits")
            self.tabular_stats = tabular_stats
        
        # Apply Z-score normalization
        for col in self.numerical_cols:
            mean = self.tabular_stats['mean'][col]
            std = self.tabular_stats['std'][col]
            self.df[col] = (self.df[col] - mean) / (std + 1e-8)

        # --- Categorical Feature Processing ---
        if self.split == 'train':
            unique_units = self.df['quantity_unit'].astype(str).unique().tolist()
            self.unit_vocab = {unit: i+1 for i, unit in enumerate(unique_units)} 
            self.unit_vocab['__unknown__'] = 0
        else:
            if unit_vocab is None:
                raise ValueError("Must provide unit_vocab for val/test splits")
            self.unit_vocab = unit_vocab
        self.unit_to_idx = self.unit_vocab

        # --- Image Transforms ---
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text) # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Text data
        text = row['item_name']
        inputs = self.tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
        
        # 2. Image data
        try:
            image_filename = Path(row['image_link']).name
            image_path = os.path.join(self.image_dir, image_filename)
            image = Image.open(image_path).convert('RGB')
            if self.split == 'train':
                pixel_values = self.train_transforms(image)
            else:
                pixel_values = self.val_transforms(image)
        except Exception:
            pixel_values = torch.zeros((3, 224, 224))

        # 3. Tabular data
        tabular_features = torch.tensor(row[self.numerical_cols].values.astype(np.float32), dtype=torch.float32)
        unit = str(row.get('quantity_unit', '__unknown__'))
        unit_idx = torch.tensor(self.unit_to_idx.get(unit, 0), dtype=torch.long)

        # 4. Price (target)
        price = torch.tensor(np.log1p(row['price']), dtype=torch.float32) if 'price' in row else torch.tensor(0, dtype=torch.float32)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'tabular_features': tabular_features,
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
    def __init__(self, unit_vocab_size, num_numerical_features, unit_embedding_dim=16, text_model_name='bert-base-uncased', image_model_name='facebook/dinov2-base'):
        super().__init__()
        
        # 1. Text encoder
        if 'roberta' in text_model_name:
            self.text_encoder = RobertaModel.from_pretrained(text_model_name)
        else:
            self.text_encoder = BertModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        # 2. Image encoder
        self.image_encoder = AutoModel.from_pretrained(image_model_name)
        image_hidden_size = self.image_encoder.config.hidden_size

        if text_hidden_size != image_hidden_size:
            self.image_proj = nn.Linear(image_hidden_size, text_hidden_size)
        else:
            self.image_proj = nn.Identity()

        # 3. Tabular tower
        self.unit_embedding = nn.Embedding(unit_vocab_size, unit_embedding_dim)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_numerical_features + unit_embedding_dim, 64),
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

    def forward(self, input_ids, attention_mask, pixel_values, tabular_features, unit_idx):
        # 1. Get text embeddings
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state

        # 2. Get image embeddings
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_embeds = self.image_proj(image_outputs.last_hidden_state)
        
        # 3. Get tabular embeddings
        unit_embeds = self.unit_embedding(unit_idx)
        tabular_inputs = torch.cat((tabular_features, unit_embeds), dim=1)
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