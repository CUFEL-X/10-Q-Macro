import akshare as ak
import pandas as pd
from datetime import datetime
import os

def fetch_and_save_data():
    # 确保目录存在
    os.makedirs("data/macro_data", exist_ok=True)
    
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # PMI 数据
    pmi = ak.macro_china_pmi()
    pmi.to_csv(f"data/macro_data/pmi.csv", index=False)
    
    # CPI 数据
    cpi = ak.macro_china_cpi()
    # PPI 数据
    ppi = ak.macro_china_ppi()
    
    # 检查列名并分别保存
    cpi.to_csv(f"data/macro_data/cpi.csv", index=False)
    ppi.to_csv(f"data/macro_data/ppi.csv", index=False)
    
    # 尝试合并 CPI 和 PPI 数据（处理列名差异）
    try:
        # 检查是否有共同列名
        common_cols = list(set(cpi.columns) & set(ppi.columns))
        if common_cols:
            cpi_ppi = pd.merge(cpi, ppi, on=common_cols[0], how="outer")
            cpi_ppi.to_csv(f"data/macro_data/cpi_ppi.csv", index=False)
        else:
            # 如果没有共同列名，分别保存
            print("No common columns found between CPI and PPI data, saving separately.")
    except Exception as e:
        print(f"Error merging CPI and PPI data: {e}")
    
    # 社会融资规模
    social_financing = ak.macro_china_shrzgm()
    social_financing.to_csv(f"data/macro_data/social_financing.csv", index=False)
    
    # 货币供应量 M0/M1/M2
    money_supply = ak.macro_china_money_supply()
    money_supply.to_csv(f"data/macro_data/money_supply.csv", index=False)
    
    # 工业增加值
    industrial_production = ak.macro_china_industrial_production_yoy()
    industrial_production.to_csv(f"data/macro_data/industrial_production.csv", index=False)
    
    # GDP 数据（季度）
    gdp = ak.macro_china_gdp()
    gdp.to_csv(f"data/macro_data/gdp.csv", index=False)
    
    # 尝试获取 MLF 操作数据
    try:
        mlf = ak.macro_china_mlf()
        mlf.to_csv(f"data/macro_data/mlf.csv", index=False)
    except AttributeError:
        print("akshare.macro_china_mlf not found, skipping MLF data.")
    
    # 尝试获取 LPR 报价数据
    try:
        lpr = ak.macro_china_lpr()
        lpr.to_csv(f"data/macro_data/lpr.csv", index=False)
    except AttributeError:
        print("akshare.macro_china_lpr not found, skipping LPR data.")
    
    # 尝试获取存款准备金率数据
    try:
        reserve_requirement_ratio = ak.macro_china_reserve_requirement_ratio()
        reserve_requirement_ratio.to_csv(f"data/macro_data/reserve_requirement_ratio.csv", index=False)
    except AttributeError:
        print("akshare.macro_china_reserve_requirement_ratio not found, skipping reserve requirement ratio data.")

if __name__ == "__main__":
    fetch_and_save_data()
    print(f"Macro data fetched and saved successfully at {datetime.now()}")
