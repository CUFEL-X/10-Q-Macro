"""模块3：基于量价的市场校准"""
import pandas as pd
from typing import Dict, List
import os

def calibrate_etf_market_conditions(
    target_date: str,
    etf_price_path: str = "data/etf/etf_2025_ohlcva.csv",
    min_avg_amount: float = 1e4,  # 从5e6调整为1e4，大幅放宽松流动性条件
    crowded_z_threshold: float = 1.5,
    lookback_days_liquidity: int = 20,
    lookback_days_crowded_short: int = 5,
    lookback_days_crowded_long: int = 120
) -> Dict[str, object]:
    """
    基于目标日期，对ETF进行市场校准。
    
    Args:
        target_date: 目标日期，格式 'YYYY-MM-DD'
        etf_price_path: ETF量价数据路径
        min_avg_amount: 最小日均成交额阈值（默认100万元）
        crowded_z_threshold: 拥挤Z-score阈值（默认1.5）
        lookback_days_*: 回看窗口参数
    
    Returns:
        dict: 包含 liquid_etfs, crowded_adjustments
    """
    # 检查文件是否存在
    if not os.path.exists(etf_price_path):
        raise FileNotFoundError(f"ETF price data file not found: {etf_price_path}")
    
    # 读取数据
    df = pd.read_csv(etf_price_path)
    
    # 转换日期列并排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 找到实际交易日
    target_date_dt = pd.to_datetime(target_date)
    actual_trade_date = df[df['date'] <= target_date_dt]['date'].max()
    
    if pd.isna(actual_trade_date):
        raise ValueError(f"No trading data found before or on {target_date}")
    
    print(f"Target date: {target_date}")
    print(f"Actual trade date: {actual_trade_date.strftime('%Y-%m-%d')}")
    
    # 按ETF代码分组
    grouped = df.groupby('code')
    
    # 初始化结果
    liquid_etfs = []
    crowded_adjustments = {}
    
    # 对每个ETF进行处理
    for ticker, group in grouped:
        # 筛选实际交易日之前的数据
        etf_data = group[group['date'] <= actual_trade_date].copy()
        etf_data = etf_data.sort_values('date')
        
        # 检查数据量
        if len(etf_data) < 15:
            print(f"Warning: ETF {ticker} has insufficient trading days ({len(etf_data)}), skipping")
            continue
        
        # 计算20日均成交额
        liquidity_data = etf_data.tail(lookback_days_liquidity)
        avg_amount = liquidity_data['amount'].mean()
        
        # 检查流动性
        if avg_amount < min_avg_amount:
            continue
        
        # 流动性合格，加入列表
        liquid_etfs.append(ticker)
        
        # 计算拥挤度
        # 取近5日成交量均值
        vol_5d_data = etf_data.tail(lookback_days_crowded_short)
        vol_5d = vol_5d_data['vol'].mean()
        
        # 取近120日成交量均值和标准差
        lookback_days = min(lookback_days_crowded_long, len(etf_data))
        if lookback_days < 60:
            print(f"Warning: ETF {ticker} has insufficient data for crowdedness calculation, using {lookback_days} days instead of {lookback_days_crowded_long}")
            continue
        
        vol_120d_data = etf_data.tail(lookback_days)
        vol_120d_mean = vol_120d_data['vol'].mean()
        vol_120d_std = vol_120d_data['vol'].std()
        
        # 计算Z-score
        if vol_120d_std > 0:
            z_score = (vol_5d - vol_120d_mean) / vol_120d_std
            
            # 判断是否拥挤
            if z_score > crowded_z_threshold:
                crowded_adjustments[ticker] = 0.75
            else:
                crowded_adjustments[ticker] = 1.0
        else:
            # 标准差为0，视为不拥挤
            crowded_adjustments[ticker] = 1.0
        
        # 折溢价监控：因缺乏IOPV数据，暂不支持
        # 相关代码已删除
    
    # 构建返回结果
    result = {
        "liquid_etfs": liquid_etfs,
        "crowded_adjustments": crowded_adjustments
    }
    
    return result

if __name__ == "__main__":
    # 测试函数
    try:
        # 降低流动性阈值来测试代码
        result = calibrate_etf_market_conditions("2025-12-31", min_avg_amount=1e5)
        print("\nMarket calibration result:")
        print(f"合格ETF数量: {len(result['liquid_etfs'])}")
        print(f"合格ETF列表: {result['liquid_etfs']}")
        # 筛选出拥挤的ETF
        crowded_etfs = [ticker for ticker, adjustment in result['crowded_adjustments'].items() if adjustment < 1.0]
        print(f"拥挤ETF: {crowded_etfs}")
        print(f"拥挤调整因子: {result['crowded_adjustments']}")
        print("折溢价监控: 因缺乏IOPV数据，暂不支持")
    except Exception as e:
        print(f"Error: {e}")
