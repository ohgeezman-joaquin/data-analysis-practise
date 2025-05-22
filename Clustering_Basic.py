import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rc('font', family='Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False

# --------------------
# 設定參數區
# --------------------
DATA_FILE = 'Online_Retail.csv'  # 請將此處更換為你的資料檔案名稱
RANDOM_STATE = 42

# --------------------
# 讀取與前處理資料
# --------------------
# 讀取 CSV 檔案
df = pd.read_csv(DATA_FILE, encoding='latin1')

# 資料預處理
# 轉換日期格式
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y/%m/%d %H:%M')

# 移除Quantity為負數或UnitPrice為0的記錄 (通常是退貨或錯誤記錄)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 確保CustomerID不是缺失值
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)

# --------------------
# 特徵工程：建立RFM模型
# --------------------
# 計算最近一次交易日期
max_date = df['InvoiceDate'].max()

# 按客戶分組計算RFM指標
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                             # Frequency
    'Quantity': 'sum',                                  # 總購買數量
    'UnitPrice': lambda x: np.sum(df.loc[x.index, 'Quantity'] * x)  # Monetary (總消費金額)
})

# 重命名欄位
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'UnitPrice': 'Monetary'
}, inplace=True)

# 確保特徵均為正數
rfm = rfm[rfm['Monetary'] > 0]

# 標準化資料
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=rfm.columns)

# --------------------
# 聚類分析
# --------------------
# 1. K-means聚類
def find_optimal_k(data, max_k=10):
    """尋找最佳的K值"""
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k+1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # 繪製肘型法則圖
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('K值')
    plt.ylabel('Inertia')
    plt.title('K-means肘型法則')
    
    # 繪製輪廓係數圖
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'o-')
    plt.xlabel('K值')
    plt.ylabel('輪廓係數')
    plt.title('K-means輪廓係數')
    
    plt.tight_layout()
    plt.show()
    
    return k_range[np.argmax(silhouette_scores)]

# 找尋最佳K值
print("正在尋找最佳K值...")
optimal_k = find_optimal_k(rfm_scaled)
print(f"最佳K值為: {optimal_k}")

# 使用最佳K值執行K-means
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
rfm['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)

# 2. DBSCAN聚類
dbscan = DBSCAN(eps=0.8, min_samples=5)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled)

# 3. 層次聚類
hc = AgglomerativeClustering(n_clusters=optimal_k)
rfm['HC_Cluster'] = hc.fit_predict(rfm_scaled)

# --------------------
# 評估聚類結果
# --------------------
print("\n========== 聚類結果評估 ==========")
# 只評估K-means和層次聚類（DBSCAN不需要預先指定族數）
for method, labels in [('KMeans', rfm['KMeans_Cluster']), ('Hierarchical', rfm['HC_Cluster'])]:
    if len(np.unique(labels)) > 1:  # 確保至少有2個聚類
        print(f"\n----- {method} 聚類評估 -----")
        print(f"輪廓係數 (Silhouette): {silhouette_score(rfm_scaled, labels):.4f}")
        print(f"Calinski-Harabasz指數: {calinski_harabasz_score(rfm_scaled, labels):.4f}")
        print(f"Davies-Bouldin指數: {davies_bouldin_score(rfm_scaled, labels):.4f}")

# --------------------
# 聚類結果視覺化
# --------------------
# 降維以便視覺化
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# 繪製聚類結果
plt.figure(figsize=(20, 5))

# K-means結果
plt.subplot(1, 3, 1)
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['KMeans_Cluster'], cmap='viridis')
plt.title('K-means聚類')
plt.xlabel('PCA維度1')
plt.ylabel('PCA維度2')

# DBSCAN結果
plt.subplot(1, 3, 2)
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['DBSCAN_Cluster'], cmap='viridis')
plt.title('DBSCAN聚類')
plt.xlabel('PCA維度1')
plt.ylabel('PCA維度2')

# 層次聚類結果
plt.subplot(1, 3, 3)
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['HC_Cluster'], cmap='viridis')
plt.title('層次聚類')
plt.xlabel('PCA維度1')
plt.ylabel('PCA維度2')

plt.tight_layout()
plt.show()

# 層次聚類樹狀圖
plt.figure(figsize=(12, 8))
Z = linkage(rfm_scaled, method='ward')
dendrogram(Z, labels=rfm.index)
plt.title('層次聚類樹狀圖')
plt.xlabel('客戶索引')
plt.ylabel('距離')
plt.show()

# --------------------
# 聚類結果分析
# --------------------
# 使用K-means結果進行分析
print("\n========== K-means聚類結果分析 ==========")
cluster_analysis = rfm.groupby('DBSCAN_Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
})

# 使用size()來計算每個聚類的客戶數
cluster_analysis['Count'] = rfm.groupby('DBSCAN_Cluster').size()
print(cluster_analysis)

# 標準化資料用於繪圖
cluster_analysis_scaled = (cluster_analysis - cluster_analysis.min()) / (cluster_analysis.max() - cluster_analysis.min())
cluster_analysis_scaled = cluster_analysis_scaled.drop('Count', axis=1)

# 轉置資料用於繪圖
cluster_analysis_scaled = cluster_analysis_scaled.T

# 繪製每個聚類的特徵
ax = cluster_analysis_scaled.plot(kind='bar', figsize=(14, 7))
ax.set_xlabel('RFM特徵')
ax.set_ylabel('標準化值')
ax.set_title('各聚類的RFM特徵比較')
ax.legend(title='聚類')

plt.tight_layout()
plt.show()

# 客戶細分結論
print("\n========== 客戶細分結論 ==========")
for label in cluster_analysis.index:
    cluster_stats = cluster_analysis.loc[label]
    print(f"聚類 {label}:")
    print(f"  客戶數: {cluster_stats['Count']:.0f}")
    print(f"  平均最近購買天數: {cluster_stats['Recency']:.2f}")
    print(f"  平均購買頻率: {cluster_stats['Frequency']:.2f}")
    print(f"  平均消費金額: {cluster_stats['Monetary']:.2f}")
    
    # 根據RFM值給出客戶分類建議
    if cluster_stats['Recency'] < cluster_analysis['Recency'].median() and cluster_stats['Monetary'] > cluster_analysis['Monetary'].median():
        print("  建議: 高價值客戶，應維持關係並提供VIP服務")
    elif cluster_stats['Recency'] < cluster_analysis['Recency'].median() and cluster_stats['Frequency'] > cluster_analysis['Frequency'].median():
        print("  建議: 忠誠客戶，可提供忠誠度獎勵")
    elif cluster_stats['Recency'] > cluster_analysis['Recency'].median() and cluster_stats['Monetary'] > cluster_analysis['Monetary'].median():
        print("  建議: 沉睡的高價值客戶，需要喚醒策略")
    elif cluster_stats['Recency'] > cluster_analysis['Recency'].median() and cluster_stats['Frequency'] < cluster_analysis['Frequency'].median():
        print("  建議: 流失風險客戶，需要挽留策略")
    print()

print("聚類分析完成！")
