import os
import pandas as pd
import re
from datetime import datetime

def create_output_dir():
    """创建输出目录"""
    output_dir = "data/processed_macro_data"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_year_month_format(date_str):
    """处理 YYYY年MM月份 格式"""
    if isinstance(date_str, str):
        match = re.match(r'(\d{4})年(\d{2})月份', date_str)
        if match:
            year, month = match.groups()
            return f"{year}-{month}-01"
    return date_str

def process_yyyymm_format(date_str):
    """处理 YYYYMM 格式"""
    if isinstance(date_str, str) or isinstance(date_str, int):
        date_str = str(date_str)
        if len(date_str) == 6:
            year = date_str[:4]
            month = date_str[4:]
            return f"{year}-{month}-01"
    return date_str

def process_quarter_format(date_str):
    """处理 YYYY年第1-4季度 格式"""
    if isinstance(date_str, str):
        # 处理 YYYY年第1-4季度 格式
        match = re.match(r'(\d{4})年第(\d)-(\d)季度', date_str)
        if match:
            year, start_q, end_q = match.groups()
            # 使用季度的第一个月作为日期
            month_map = {'1': '01', '2': '04', '3': '07', '4': '10'}
            month = month_map.get(start_q, '01')
            return f"{year}-{month}-01"
        # 处理 YYYY年第1季度 格式
        match = re.match(r'(\d{4})年第(\d)季度', date_str)
        if match:
            year, q = match.groups()
            # 使用季度的第一个月作为日期
            month_map = {'1': '01', '2': '04', '3': '07', '4': '10'}
            month = month_map.get(q, '01')
            return f"{year}-{month}-01"
    return date_str

def process_chinese_date_format(date_str):
    """处理 YYYY年MM月DD日 格式"""
    if isinstance(date_str, str):
        match = re.match(r'(\d{4})年(\d{2})月(\d{2})日', date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
    return date_str

def process_file(input_file, output_dir):
    """处理单个文件"""
    print(f"Processing file: {input_file}")
    
    # 读取文件
    df = pd.read_csv(input_file)
    print(f"Original columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")
    
    # 获取文件名
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, filename)
    
    # 根据文件名处理不同的时间格式
    if filename == "pmi.csv" or filename == "cpi.csv" or filename == "ppi.csv" or filename == "money_supply.csv" or filename == "cpi_ppi.csv":
        # 处理 月份 列
        if "月份" in df.columns:
            df["date"] = df["月份"].apply(process_year_month_format)
            df = df.drop("月份", axis=1)
    
    elif filename == "social_financing.csv":
        # 处理 月份 列
        if "月份" in df.columns:
            df["date"] = df["月份"].apply(process_yyyymm_format)
            df = df.drop("月份", axis=1)
    
    elif filename == "industrial_production.csv":
        # 处理 日期 列
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "date"})
    
    elif filename == "gdp.csv":
        # 处理 季度 列
        if "季度" in df.columns:
            df["date"] = df["季度"].apply(process_quarter_format)
            # 再次处理，确保所有格式都被转换
            df["date"] = df["date"].apply(lambda x: process_year_month_format(x) if isinstance(x, str) and "年第" in x else x)
            df = df.drop("季度", axis=1)
    
    elif filename == "lpr.csv":
        # 处理 TRADE_DATE 列
        if "TRADE_DATE" in df.columns:
            df = df.rename(columns={"TRADE_DATE": "date"})
    
    elif filename == "reserve_requirement_ratio.csv":
        # 处理 公布时间 列
        if "公布时间" in df.columns:
            df["date"] = df["公布时间"].apply(process_chinese_date_format)
            # 保留生效时间列，但不重命名
    
    # 保存处理后的文件
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to: {output_file}")
    print(f"Processed columns: {list(df.columns)}")
    print(f"First few rows after processing:\n{df.head()}")
    print("=" * 50)

def process_all_files():
    """处理所有文件"""
    input_dir = "data/macro_data"
    output_dir = create_output_dir()
    
    # 遍历所有CSV文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_dir, filename)
            process_file(input_file, output_dir)
    
    print("All files processed successfully!")

if __name__ == "__main__":
    process_all_files()
