import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, median_absolute_error,
                             explained_variance_score,
                             mean_absolute_percentage_error)

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

# 模型設定
RANDOM_STATE = 42

# --------------------
# 讀取與前處理資料
# --------------------
# 讀取 CSV 檔案 (以分號作為分隔符)
data = pd.read_csv(DATA_FILE, sep=';')

# 分離特徵與目標
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN].astype(float)  # 目標為數值型變數

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
# 模型建立：選用常見的回歸模型
# --------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

# 建立模型字典，並新增多項式回歸（二次多項式）模型
models = {
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=RANDOM_STATE),
    'Random Forest Regressor': RandomForestRegressor(random_state=RANDOM_STATE),
    'SVR': SVR(),  # 預設 kernel 為 'rbf'
    'Linear Regression': LinearRegression(),
    'Neural Network Regressor': MLPRegressor(max_iter=1000, random_state=RANDOM_STATE),
    'Polynomial Regression (degree=2)': Pipeline(steps=[
         ('preprocessor', preprocessor),
         ('poly', PolynomialFeatures(degree=2)),
         ('regressor', LinearRegression())
    ])
}

# --------------------
# 切分資料集
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# --------------------
# 模型訓練與評估
# --------------------
print("Target Variable Statistics:")
print(data[TARGET_COLUMN].describe())
for model_name, model in models.items():
    print(f"========== {model_name} ==========")
    # 如果模型已經是 Pipeline（如多項式回歸的情況），則直接使用；否則將 preprocessor 與 regressor 組合進 Pipeline 中
    if not isinstance(model, Pipeline):
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    else:
        pipeline = model
    
    # 模型訓練
    pipeline.fit(X_train, y_train)

    # 模型預測
    y_pred = pipeline.predict(X_test)
    
    # 計算各項回歸指標
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 顯示目標變數統計摘要
    print("\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Median Absolute Error: {medae:.4f}")
    print(f"Explained Variance Score: {explained_variance:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")
    print()

# -------------------------------
# 10. 交叉驗證評估
# -------------------------------
print("\n========== 交叉驗證評估 ==========")
# 定義兩種交叉驗證方法
cv_methods = {
    'KFold': KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
}

for cv_name, cv in cv_methods.items():
    print(f"\n----- 使用 {cv_name} -----")
    for model_name, model in models.items():
        # 如果 model 本身就是 Pipeline，就直接用它；否則才外層包 preprocessor
        if isinstance(model, Pipeline):
            cv_pipeline = model
        else:
            cv_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        mse_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
        print(f"{model_name}:")
        print(f"  R^2: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  MSE: {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")