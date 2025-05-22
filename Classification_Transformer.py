import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# --------------------
# 設定參數區
# --------------------
DATA_FILE = 'bank_data.csv'  # 請更換為你的檔案
TARGET_COLUMN = 'y'
# 以下特徵需自行調整，假設我們只取一部分作範例（實際應用時可以全部使用）
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
NUMERIC_FEATURES = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
RANDOM_STATE = 42

# --------------------
# 讀取與前處理資料
# --------------------
data = pd.read_csv(DATA_FILE)

# 分離特徵與目標
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# 目標變數編碼：'yes'->1, 'no'->0
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# 處理數值資料
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[NUMERIC_FEATURES])
# 將數值資料轉換成 float32 tensor
X_numeric = torch.tensor(X_numeric, dtype=torch.float32)

# 處理類別資料：使用 LabelEncoder 將各欄位轉為整數
X_cat = X[CATEGORICAL_FEATURES].copy()
encoders = {}
cat_dims = []
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col])
    encoders[col] = le
    # 紀錄每個欄位的類別數（作為 embedding 的輸入維度）
    cat_dims.append(len(le.classes_))

# 將類別資料轉換成 long tensor
X_cat = torch.tensor(X_cat.values, dtype=torch.long)

# 目標轉為 tensor
y = torch.tensor(y, dtype=torch.long)

# 切分資料集（這裡直接對 tensor 切分）
num_samples = X_numeric.shape[0]
indices = np.arange(num_samples)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y.numpy())

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
# 定義 TabTransformer 模型
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
            nn.Linear(64, 2)  # 二元分類
        )
    
    def forward(self, numeric, categorical):
        # categorical shape: (batch, n_cat)
        # 取每個類別特徵的 embedding，得到一個 list，元素 shape (batch, embed_dim)
        cat_embeds = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        # 合併成 tensor: (batch, n_cat, embed_dim)
        cat_embeds = torch.stack(cat_embeds, dim=1)
        # Transformer 需要輸入 shape (seq_len, batch, embed_dim)，這裡 seq_len = n_cat
        cat_embeds = cat_embeds.permute(1, 0, 2)
        # 經過 transformer encoder，輸出 shape 同上
        transformer_out = self.transformer(cat_embeds)
        # 將 transformer 輸出做平均池化，結果 shape: (batch, embed_dim)
        cat_out = transformer_out.mean(dim=0)
        # 結合數值特徵 (batch, num_numeric) 與類別特徵 (batch, embed_dim)
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for numeric_batch, cat_batch, labels in train_loader:
        numeric_batch = numeric_batch.to(device)
        cat_batch = cat_batch.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(numeric_batch, cat_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * numeric_batch.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

# --------------------
# 模型評估
# --------------------
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for numeric_batch, cat_batch, labels in test_loader:
        numeric_batch = numeric_batch.to(device)
        cat_batch = cat_batch.to(device)
        labels = labels.to(device)
        outputs = model(numeric_batch, cat_batch)
        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 計算測試集準確率
test_accuracy = accuracy_score(all_labels, all_preds)
print("測試集準確率: {:.4f}".format(test_accuracy))

# 輸出詳細的分類報告
report = classification_report(all_labels, all_preds, digits=4)
print(report)