import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --------------------
# 設定參數區
# --------------------
DATA_FILE = 'bank_data.csv'  # 請將此處更換為你的資料檔案名稱

# 欄位設定：目標欄位為 y，其餘為特徵
TARGET_COLUMN = 'y'
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
NUMERIC_FEATURES = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# 模型設定
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
y = le.fit_transform(y)  # 例如: 'no' -> 0, 'yes' -> 1

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
# 模型建立
# --------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'SVM': SVC(random_state=RANDOM_STATE),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)
}

# --------------------
# 切分資料集
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# # --------------------
# # 模型訓練與評估
# # --------------------
# for model_name, model in models.items():
#     print(f"========== {model_name} ==========")
#     # 建立整合前處理與分類器的 Pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])
    
#     # 模型訓練
#     pipeline.fit(X_train, y_train)
    
#     # 模型預測
#     y_pred = pipeline.predict(X_test)
    
#     # 模型評估
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")
#     print(classification_report(y_test, y_pred))

# # -------------------------------
# # 10. 交叉驗證評估
# # -------------------------------
# print("\n========== 交叉驗證評估 ==========")
# # 定義兩種交叉驗證方法
# cv_methods = {
#     'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
#     'KFold': KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# }

# for cv_name, cv in cv_methods.items():
#     print(f"\n----- 使用 {cv_name} -----")
#     for model_name, model in models.items():
#         pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', model)
#         ])
#         scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
#         print(f"{model_name} Accuracy ({cv_name}): {np.mean(scores):.4f} (± {np.std(scores):.4f})")

# # -------------------------------
# # 11.相關性分析（Correlation）
# # -------------------------------
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 只選數值欄位
# num_df = data[NUMERIC_FEATURES]
# corr = num_df.corr()
# print(corr)

# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Numeric Features Correlation Matrix")
# plt.show()

# # -------------------------------
# # 12.特徵重要性（Feature Importance）
# # -------------------------------
# # 使用隨機森林模型來計算特徵重要性
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import matplotlib.pyplot as plt

# # 重新建立並訓練 Random Forest pipeline（或是直接用你先前訓練好的 pipeline 變數）
# rf_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
# ])
# rf_pipeline.fit(X_train, y_train)

# # 取出 One-Hot 之後的欄位名稱
# onehot_cols = rf_pipeline.named_steps['preprocessor'] \
#     .named_transformers_['cat'] \
#     .named_steps['onehot'] \
#     .get_feature_names_out(CATEGORICAL_FEATURES)
# all_features = np.hstack([NUMERIC_FEATURES, onehot_cols])

# # 抓出 feature_importances_ 並排序
# importances = rf_pipeline.named_steps['classifier'].feature_importances_
# indices = np.argsort(importances)[::-1]

# # 列印前 10 名重要特徵
# print("\n===== Random Forest 前 10 名重要特徵 =====")
# for idx in indices[:10]:
#     print(f"{all_features[idx]}: {importances[idx]:.4f}")

# # 繪製所有特徵重要度長條圖
# plt.figure(figsize=(12,6))
# plt.title("Feature Importances by Random Forest")
# plt.bar(range(len(importances)), importances[indices], align="center")
# plt.xticks(range(len(importances)), all_features[indices], rotation=90)
# plt.xlabel("Feature")
# plt.ylabel("Importance")
# plt.tight_layout()
# plt.show()

# -------------------------------
# 13.pandas_profiling
# -------------------------------
import pandas as pd
from ydata_profiling import ProfileReport   # 若是舊版則 from pandas_profiling import ProfileReport

# 2. 讀資料
df = pd.read_csv('bank_data.csv')

# 3. 建立報告
profile = ProfileReport(df,
                        title="Bank Data Profiling Report",
                        explorative=True)

# 4. 顯示於 Notebook
# profile.to_notebook_iframe()

# 5. 或輸出成 HTML 檔
profile.to_file("bank_data_profile.html")
