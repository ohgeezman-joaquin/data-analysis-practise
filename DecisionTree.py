import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# --------------------
# 設定參數區
# --------------------
DATA_FILE = 'bank_data.csv'  # 請將此處更換為你的資料檔案名稱
TARGET_COLUMN = 'y'
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
NUMERIC_FEATURES = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
RANDOM_STATE = 42

# --------------------
# 讀取與前處理資料
# --------------------
# 讀取 CSV 檔案
data = pd.read_csv(DATA_FILE)

# 分離特徵與目標
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# 目標變數編碼：將 'yes' 與 'no' 轉換為 1 與 0
le = LabelEncoder()
y = le.fit_transform(y)

# 建立前處理流程：數值資料標準化；類別資料做 One-Hot 編碼
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ]
)

# --------------------
# 切分資料集
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# --------------------
# 建立 DecisionTreeClassifier 與 GridSearchCV
# --------------------
# 建立包含前處理與分類器的 Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

# 設定 GridSearchCV 的參數範圍
param_grid = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# --------------------
# 輸出最佳參數與評估結果
# --------------------
print("最佳參數:", grid_search.best_params_)
print("最佳交叉驗證準確率: {:.4f}".format(grid_search.best_score_))

# 在測試集上評估最佳模型
y_pred = grid_search.predict(X_test)
print("測試集準確率: {:.4f}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
