import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, median_absolute_error,
                             explained_variance_score,
                             mean_absolute_percentage_error)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --------------------
# 設定參數區
# --------------------
DATA_FILE = 'student_por.csv'  # 修改成你的資料檔案名稱，請確認資料的分隔符為分號 (;)
TARGET_COLUMN = 'G3'
# 根據資料描述，將特徵分為類別特徵與數值特徵：
CATEGORICAL_FEATURES = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 
    'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
    'famsup', 'paid', 'activities', 'nursery', 'higher', 
    'internet', 'romantic'
]

NUMERIC_FEATURES = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
    'failures', 'famrel', 'freetime', 'goout', 
    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
]
RANDOM_STATE = 42

# --------------------
# 讀取與前處理資料
# --------------------
# data = pd.read_csv(DATA_FILE)
data = pd.read_csv(DATA_FILE, sep=';')

# 分離特徵與目標（這裡假設目標變數已為連續值，不需要進行 LabelEncoder）
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# 處理數值資料：標準化後轉為 float32 tensor
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[NUMERIC_FEATURES])
X_numeric = torch.tensor(X_numeric, dtype=torch.float32)

# 處理類別資料：使用 LabelEncoder 將各欄位轉為整數
# 注意：這裡僅為處理類別特徵用於建立 embedding，不會對回歸目標做編碼
X_cat = X[CATEGORICAL_FEATURES].copy()
encoders = {}
cat_dims = []
from sklearn.preprocessing import LabelEncoder
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col])
    encoders[col] = le
    # 紀錄每個欄位的類別數（作為 embedding 的輸入維度）
    cat_dims.append(len(le.classes_))
X_cat = torch.tensor(X_cat.values, dtype=torch.long)

# 目標轉為 tensor（如果 y 是 Series，可使用 values 屬性）
y = torch.tensor(y.values, dtype=torch.float32)
# 為符合 MSELoss 計算需求，將 y 轉換成 (N, 1) 的張量
y = y.view(-1, 1)

# 切分資料集（直接對 tensor 進行切分）
num_samples = X_numeric.shape[0]
indices = np.arange(num_samples)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)

X_numeric_train = X_numeric[train_idx]
X_cat_train = X_cat[train_idx]
y_train = y[train_idx]

X_numeric_test = X_numeric[test_idx]
X_cat_test = X_cat[test_idx]
y_test = y[test_idx]

# 建立 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_numeric_train, X_cat_train, y_train)
test_dataset = TensorDataset(X_numeric_test, X_cat_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --------------------
# 定義 TabTransformer 模型（回歸版本）
# --------------------
class TabTransformer(nn.Module):
    def __init__(self, num_numeric, cat_dims, embed_dim=8, n_transformer_layers=2, n_heads=2, dropout=0.1):
        """
        num_numeric: 數值特徵數量
        cat_dims: 每個類別特徵的類別數列表
        embed_dim: 每個類別特徵的 embedding 維度
        """
        super(TabTransformer, self).__init__()
        self.num_numeric = num_numeric
        self.n_cat = len(cat_dims)
        # 為每個類別特徵建立 embedding
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
        
        # Transformer Encoder 處理類別資料
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # 最終全連接層：結合數值特徵與類別特徵（透過 transformer 輸出）
        self.fc = nn.Sequential(
            nn.Linear(num_numeric + embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 輸出一個連續值
        )
    
    def forward(self, numeric, categorical):
        # categorical shape: (batch, n_cat)
        # 每個類別特徵經過 embedding 之後得到 (batch, embed_dim)
        cat_embeds = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        # 合併成 tensor: (batch, n_cat, embed_dim)
        cat_embeds = torch.stack(cat_embeds, dim=1)
        # Transformer Encoder 需要輸入 shape 為 (seq_len, batch, embed_dim)
        cat_embeds = cat_embeds.permute(1, 0, 2)
        transformer_out = self.transformer(cat_embeds)
        # 對 Transformer 輸出做平均池化，得到 (batch, embed_dim)
        cat_out = transformer_out.mean(dim=0)
        # 結合數值資料 (batch, num_numeric) 與類別特徵 (batch, embed_dim)
        x = torch.cat([numeric, cat_out], dim=1)
        out = self.fc(x)
        return out

# 建立模型實例
num_numeric = len(NUMERIC_FEATURES)
model = TabTransformer(num_numeric=num_numeric, cat_dims=cat_dims, embed_dim=8, n_transformer_layers=2, n_heads=2, dropout=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --------------------
# 訓練模型
# --------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10000

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for numeric_batch, cat_batch, targets in train_loader:
        numeric_batch = numeric_batch.to(device)
        cat_batch = cat_batch.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(numeric_batch, cat_batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * numeric_batch.size(0)
    train_loss /= len(train_loader.dataset)
    if (epoch+1) % 50 == 0 or epoch==0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

# --------------------
# 模型評估
# --------------------
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for numeric_batch, cat_batch, targets in test_loader:
        numeric_batch = numeric_batch.to(device)
        cat_batch = cat_batch.to(device)
        targets = targets.to(device)
        outputs = model(numeric_batch, cat_batch)
        all_preds.extend(outputs.cpu().numpy().squeeze())
        all_targets.extend(targets.cpu().numpy().squeeze())

# 計算測試集的 MSE 與 R-squared
test_mse = mean_squared_error(all_targets, all_preds)
test_r2 = r2_score(all_targets, all_preds)
test_mae = mean_absolute_error(all_targets, all_preds)
test_median_ae = median_absolute_error(all_targets, all_preds)
test_explained_var = explained_variance_score(all_targets, all_preds)
test_mape = mean_absolute_percentage_error(all_targets, all_preds)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R-squared: {test_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Median Absolute Error: {test_median_ae:.4f}")
print(f"Test Explained Variance Score: {test_explained_var:.4f}")
print(f"Test MAPE: {test_mape:.4f}")
print("訓練完成！")
