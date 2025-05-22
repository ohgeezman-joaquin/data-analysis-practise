import mysql.connector
import pandas as pd

# 設定 MySQL 連線參數，請根據實際情況修改
DB_CONFIG  = {
    'host': 'localhost',          # 資料庫主機位址
    'user': '',      # 資料庫使用者名稱
    'password': '',  # 資料庫密碼
    'database': ''   # 目標資料庫名稱
}
TABLE_NAME = 'bank_data'         # 欲匯出的資料表名稱
CSV_FILENAME = f"{TABLE_NAME}.csv"  # 輸出 CSV 檔案名稱，沿用資料表名稱

def convert_mysql_to_csv(db_config, table_name, csv_filename):
    """
    連接 MySQL 資料庫並將指定資料表的資料轉成 CSV 檔案。
    
    Args:
        db_config (dict): 資料庫連線設定參數。
        table_name (str): 欲匯出的資料表名稱。
        csv_filename (str): 輸出 CSV 檔案名稱。
    """
    try:
        # 建立資料庫連線
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 執行查詢
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        rows = cursor.fetchall()

        # 取得欄位名稱
        columns = [desc[0] for desc in cursor.description]

        # 利用 pandas 建立 DataFrame 並轉存成 CSV 檔案
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"資料庫 {table_name} 內容已轉換成 CSV 檔案：{csv_filename}")

    except mysql.connector.Error as err:
        print("資料庫連線或操作發生錯誤：", err)
    finally:
        # 確保關閉連線
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    convert_mysql_to_csv(DB_CONFIG, TABLE_NAME, CSV_FILENAME)
