# 匯入需要的函式庫
import pandas as pd           # 資料處理
import numpy as np            # 數值運算
import matplotlib.pyplot as plt  # 基本繪圖
import seaborn as sns         # 資料視覺化

# 控制各圖是否產生的開關
PLOT_HEAD = True
PLOT_BASIC_INFO = False
PLOT_NUMERICAL = False
PLOT_CATEGORICAL = False
PLOT_HEATMAP = False
PLOT_PAIRPLOT = False
PLOT_EDA_REPORT = False

# -------------------------------
# 1. 讀取資料
# -------------------------------
df = pd.read_csv("student_por.csv", delimiter=';')
if PLOT_HEAD:
    print("資料前五筆：")
    print(df.head())

# -------------------------------
# 2. 了解資料基本資訊
# -------------------------------
if PLOT_BASIC_INFO:
    print("\n資料形狀 (列數, 欄數):", df.shape)
    print("\n資料摘要：")
    print(df.info())
    print("\n數值型欄位描述統計：")
    print(df.describe())

# -------------------------------
# 4. 數值資料分佈分析
# -------------------------------
numerical_cols = ['age', 'traveltime', 'studytime', 'failures', 
                  'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 
                  'health', 'absences', 'G1', 'G2', 'G3']

if PLOT_NUMERICAL:
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=10, color='skyblue')
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("frequency")
        plt.show()

# -------------------------------
# 5. 類別資料計數分析
# -------------------------------
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                    'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                    'famsup', 'paid', 'activities', 'nursery', 'higher', 
                    'internet', 'romantic']

if PLOT_CATEGORICAL:
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        order = df[col].value_counts().index
        sns.countplot(x=col, hue=col, data=df, order=order, palette="Set2")
        plt.title(f"{col} count")
        plt.xlabel(col)
        plt.ylabel("number")
        plt.xticks(rotation=45)
        plt.show()

# -------------------------------
# 6. 數值變數間相關性分析
# -------------------------------
if PLOT_HEATMAP:
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Numerical Variables") #數值變數相關係數矩陣
    plt.show()

# -------------------------------
# 7. 數值變數之間的配對圖（Pairplot）
# -------------------------------
if PLOT_PAIRPLOT:
    sns.pairplot(df[numerical_cols], diag_kind='kde')
    plt.suptitle("numerical variable Pairplot", y=1.02)
    plt.show()

# -------------------------------
# 8. 生成自動 EDA 報告（選用）
# -------------------------------
if PLOT_EDA_REPORT:
    try:
        from pandas_profiling import ProfileReport
        profile = ProfileReport(df, title="學生成績資料 EDA 報告", explorative=True)
        profile.to_file("eda_report.html")
        print("自動 EDA 報告已生成，請檢查 eda_report.html 檔案")
    except ImportError:
        print("pandas_profiling 套件尚未安裝，如需自動報告請先執行：pip install ydata-profiling")
