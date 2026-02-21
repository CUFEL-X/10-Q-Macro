"""模块1：宏观状态与景气度"""
import pandas as pd
import os
from typing import Dict, Any

def detect_macro_regime_and_score(
    target_date: str,
    macro_data_dir: str = "data/processed_macro_data",
    fallback_to_last: bool = True
) -> Dict[str, Any]:
    """
    基于目标日期，识别当前宏观状态并生成景气度评分。
    
    Args:
        target_date: 目标日期，格式 'YYYY-MM-DD' 或 'YYYY-MM'
        macro_data_dir: 处理后的宏观数据目录
        fallback_to_last: 若最新数据缺失，是否回退到历史最新
    
    Returns:
        dict: 包含 regime, momentum, score 等字段
    """
    # 转换目标日期为年月格式
    if len(target_date) >= 10:
        # YYYY-MM-DD 格式
        target_year_month = target_date[:7]
    else:
        # YYYY-MM 格式
        target_year_month = target_date
    
    # 读取并处理 PMI 数据
    pmi_file = os.path.join(macro_data_dir, "pmi.csv")
    if not os.path.exists(pmi_file):
        raise FileNotFoundError(f"PMI data file not found: {pmi_file}")
    
    pmi_df = pd.read_csv(pmi_file)
    pmi_df['date'] = pd.to_datetime(pmi_df['date'])
    pmi_df['year_month'] = pmi_df['date'].dt.strftime('%Y-%m')
    pmi_df = pmi_df.sort_values('date', ascending=False)
    
    # 读取并处理 CPI 数据
    cpi_file = os.path.join(macro_data_dir, "cpi.csv")
    if not os.path.exists(cpi_file):
        raise FileNotFoundError(f"CPI data file not found: {cpi_file}")
    
    cpi_df = pd.read_csv(cpi_file)
    cpi_df['date'] = pd.to_datetime(cpi_df['date'])
    cpi_df['year_month'] = cpi_df['date'].dt.strftime('%Y-%m')
    cpi_df = cpi_df.sort_values('date', ascending=False)
    
    # 读取并处理 PPI 数据
    ppi_file = os.path.join(macro_data_dir, "ppi.csv")
    if not os.path.exists(ppi_file):
        raise FileNotFoundError(f"PPI data file not found: {ppi_file}")
    
    ppi_df = pd.read_csv(ppi_file)
    ppi_df['date'] = pd.to_datetime(ppi_df['date'])
    ppi_df['year_month'] = ppi_df['date'].dt.strftime('%Y-%m')
    ppi_df = ppi_df.sort_values('date', ascending=False)
    
    # 读取并处理工业增加值数据
    ip_file = os.path.join(macro_data_dir, "industrial_production.csv")
    if not os.path.exists(ip_file):
        raise FileNotFoundError(f"Industrial production data file not found: {ip_file}")
    
    ip_df = pd.read_csv(ip_file)
    ip_df['date'] = pd.to_datetime(ip_df['date'])
    ip_df['year_month'] = ip_df['date'].dt.strftime('%Y-%m')
    ip_df = ip_df.sort_values('date', ascending=False)
    
    # 找到不晚于目标日期的最新数据
    def get_latest_data(df, target_ym, fallback):
        # 筛选不晚于目标日期的数据
        filtered = df[df['year_month'] <= target_ym]
        if filtered.empty:
            if fallback and not df.empty:
                # 回退到历史最新
                return df.iloc[0]
            else:
                raise ValueError(f"No data found for target date or earlier: {target_ym}")
        return filtered.iloc[0]
    
    # 获取各指标的最新数据
    latest_pmi = get_latest_data(pmi_df, target_year_month, fallback_to_last)
    latest_cpi = get_latest_data(cpi_df, target_year_month, fallback_to_last)
    latest_ppi = get_latest_data(ppi_df, target_year_month, fallback_to_last)
    latest_ip = get_latest_data(ip_df, target_year_month, fallback_to_last)
    
    # 提取关键值
    pmi = latest_pmi['制造业-指数']
    cpi_yoy = latest_cpi['全国-同比增长']
    ppi_yoy = latest_ppi['当月同比增长']
    
    # 处理工业增加值可能的缺失值
    industrial_yoy = latest_ip['今值']
    if pd.isna(industrial_yoy):
        # 如果工业增加值为NaN，使用PMI作为替代
        industrial_yoy = pmi - 45.0
        print("Warning: Industrial production data is NaN, using PMI as substitute")
    
    # 实际使用的年月
    actual_year_month = latest_pmi['year_month']
    
    # 计算四象限状态
    growth = (pmi > 50.0) or (industrial_yoy > 5.0)
    inflation = (cpi_yoy > 2.5) or (ppi_yoy > 0.0)
    
    if growth and inflation:
        regime = "overheat"
    elif growth and not inflation:
        regime = "recovery"
    elif not growth and inflation:
        regime = "stagflation"
    else:
        regime = "recession"
    
    # 计算动量（近3个月变化）
    def calculate_momentum(df, value_col, months=3):
        # 获取最近几个月的数据
        recent_data = df.head(months)
        if len(recent_data) < 2:
            return 0.0
        
        # 计算简单差值（最新值 - 最早值）
        latest = recent_data.iloc[0][value_col]
        earliest = recent_data.iloc[-1][value_col]
        momentum = (latest - earliest) / months
        
        # 标准化到合理范围
        return momentum / 10.0
    
    growth_momentum = calculate_momentum(pmi_df, '制造业-指数')
    inflation_momentum = calculate_momentum(cpi_df, '全国-同比增长')
    
    # 生成 equity_friendly_score
    # 基础分 = PMI 映射到 [0,1]
    base_score = max(0.0, min(1.0, (pmi - 45.0) / 10.0))
    
    # 动量加分
    momentum_bonus = 0.1 * growth_momentum  # 假设 momentum ∈ [-0.5, 0.5]
    
    # 通缩惩罚（PPI长期负）
    deflation_penalty = 0.1 if ppi_yoy < -2.0 else 0.0
    
    score = base_score + momentum_bonus - deflation_penalty
    score = max(0.0, min(1.0, score))
    
    # 构建返回结果
    result = {
        "as_of_date": actual_year_month,
        "regime": regime,
        "growth_momentum": growth_momentum,
        "inflation_momentum": inflation_momentum,
        "equity_friendly_score": score,
        "raw_indicators": {
            "pmi": pmi,
            "cpi_yoy": cpi_yoy,
            "ppi_yoy": ppi_yoy,
            "industrial_yoy": industrial_yoy
        }
    }
    
    return result

if __name__ == "__main__":
    # 测试函数
    test_date = "2025-12-31"
    try:
        result = detect_macro_regime_and_score(test_date)
        print("宏观状态与景气度分析结果:")
        print(f"参考日期: {result['as_of_date']}")
        print(f"宏观状态: {result['regime']}")
        print(f"增长动量: {result['growth_momentum']:.4f}")
        print(f"通胀动量: {result['inflation_momentum']:.4f}")
        print(f"权益友好度评分: {result['equity_friendly_score']:.4f}")
        print("原始指标:")
        for key, value in result['raw_indicators'].items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
