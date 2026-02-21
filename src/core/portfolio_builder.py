"""模块4：ETF组合构建"""
import pandas as pd
import os
from typing import Dict, Any

# 四象限资产配置基础权重
REGIME_BASE_WEIGHTS = {
    "recovery": {"stock": 0.7, "bond": 0.2, "commodity": 0.1},
    "overheat": {"stock": 0.6, "bond": 0.1, "commodity": 0.3},
    "stagflation": {"stock": 0.3, "bond": 0.3, "commodity": 0.4},
    "recession": {"stock": 0.4, "bond": 0.5, "commodity": 0.1}
}

def build_portfolio(
    macro_state: dict,
    market_cond: dict,
    current_themes: list = None,
    etf_meta_path: str = "data/etf/processed_etf_basic.csv"
) -> dict:
    """
    构建ETF投资组合。
    
    Args:
        macro_state: 来自 src/core/macro_regime.py 的输出
        market_cond: 来自 src/core/market_calibration.py 的输出
        current_themes: 当前推荐的多个主题（top_5_themes）
        etf_meta_path: ETF基本信息路径（相对于项目根目录）
    
    Returns:
        dict: 包含 target_weights 和 reasons
    """
    # 定义允许的主题列表
    ALLOWED_THEMES = [
        "人工智能", "半导体", "新能源车", "光伏", "医药", "消费",
        "金融", "周期", "高端制造", "数字经济", "科技创新", "新材料",
        "生物医药", "通用航空", "国企改革", "红利低波", "农业",
        "公用事业", "电力", "低碳经济", "碳中和", "物联网", "教育",
        "养老产业", "ESG", "可持续发展", "传媒", "房地产", "物流"
    ]
    # 检查文件是否存在
    if not os.path.exists(etf_meta_path):
        raise FileNotFoundError(f"ETF basic info file not found: {etf_meta_path}")
    
    # 读取ETF基本信息
    etf_df = pd.read_csv(etf_meta_path)
    
    # 确保code为字符串类型
    etf_df['code'] = etf_df['code'].astype(str)
    
    # 提取输入参数
    regime = macro_state.get('regime')
    equity_friendly_score = macro_state.get('equity_friendly_score', 0.5)
    liquid_etfs = market_cond.get('liquid_etfs', [])
    crowded_adjustments = market_cond.get('crowded_adjustments', {})
    
    # 确保liquid_etfs中的code为字符串
    liquid_etfs = [str(code) for code in liquid_etfs]
    
    # 过滤ETF数据，只保留流动性好的ETF
    liquid_etf_df = etf_df[etf_df['code'].isin(liquid_etfs)].copy()
    
    # 1. 确定大类资产基础权重
    if regime not in REGIME_BASE_WEIGHTS:
        raise ValueError(f"Unknown regime: {regime}")
    
    base_weights = REGIME_BASE_WEIGHTS[regime]
    
    # 动态调节股票权重
    base_stock_weight = base_weights['stock']
    adjusted_stock_weight = base_stock_weight * (0.8 + 0.4 * equity_friendly_score)
    
    # 计算其他资产的权重调整比例
    total_non_stock = base_weights['bond'] + base_weights['commodity']
    if total_non_stock > 0:
        bond_ratio = base_weights['bond'] / total_non_stock
        commodity_ratio = base_weights['commodity'] / total_non_stock
        
        adjusted_bond_weight = (1 - adjusted_stock_weight) * bond_ratio
        adjusted_commodity_weight = (1 - adjusted_stock_weight) * commodity_ratio
    else:
        adjusted_bond_weight = 0
        adjusted_commodity_weight = 0
    
    # 2. 筛选宽基ETF
    broad_etfs = liquid_etf_df[
        (liquid_etf_df['asset_class'] == '股票') & 
        (liquid_etf_df['theme'] == '宽基')
    ]['code'].tolist()
    
    # 若无匹配，回退到默认 "510300"
    broad_etf = broad_etfs[0] if broad_etfs else "510300"
    if not broad_etfs:
        print(f"Warning: No broad ETF found, falling back to default: {broad_etf}")
    
    # 3. 筛选主题ETF
    theme_etfs = []
    # 收集所有候选主题
    candidate_themes = []
    if current_themes:
        candidate_themes.extend([theme['theme'] for theme in current_themes])
    # 去重
    candidate_themes = list(set(candidate_themes))
    
    if candidate_themes:
        # 筛选主题ETF，根据tag_analysis.txt的标签结构进行匹配
        for theme in candidate_themes:
            if theme in ALLOWED_THEMES:
                # 匹配逻辑：
                # 1. 资产大类为股票
                # 2. theme字段为"主题"且sub_theme字段匹配
                # 3. 或者theme字段直接匹配（有些ETF可能直接将主题放在theme字段）
                # 4. 或者sub_theme字段匹配（有些ETF可能将主题放在sub_theme字段）
                matched_etfs = liquid_etf_df[
                    (liquid_etf_df['asset_class'] == '股票') & 
                    ((
                        (liquid_etf_df['theme'] == '主题') & 
                        (liquid_etf_df['sub_theme'] == theme)
                    ) | (
                        liquid_etf_df['theme'] == theme
                    ) | (
                        liquid_etf_df['sub_theme'] == theme
                    ))
                ]['code'].tolist()
                theme_etfs.extend(matched_etfs)
        # 去重
        theme_etfs = list(set(theme_etfs))
        
        # 打印匹配结果
        if theme_etfs:
            print(f"   匹配到的主题ETF数量: {len(theme_etfs)}")
            print(f"   匹配到的主题ETF代码: {theme_etfs[:5]}{'...' if len(theme_etfs) > 5 else ''}")
        else:
            print(f"   未匹配到主题ETF，将使用宽基ETF")
    
    # 4. 分配债券与商品ETF
    bond_etfs = liquid_etf_df[liquid_etf_df['asset_class'] == '债券']['code'].tolist()
    commodity_etfs = liquid_etf_df[liquid_etf_df['asset_class'] == '商品']['code'].tolist()
    
    # 债券ETF选择
    bond_etf = bond_etfs[0] if bond_etfs else "511260"
    
    # 检查债券ETF是否在liquid_etfs中
    bond_available = bond_etf in liquid_etfs
    if not bond_available:
        print(f"Warning: Bond ETF {bond_etf} not in liquid ETFs, skipping")
        adjusted_bond_weight = 0
    
    # 商品ETF等权配置
    commodity_available = len(commodity_etfs) > 0
    if not commodity_available:
        print(f"Warning: No commodity ETFs available, using default 518880")
        commodity_etfs = ["518880"]  # 默认使用黄金ETF
    
    # 商品ETF等权分配
    commodity_weight_per_etf = adjusted_commodity_weight / len(commodity_etfs)
    print(f"Commodity ETFs: {commodity_etfs}")
    print(f"Equal weight per commodity ETF: {commodity_weight_per_etf:.4f}")
    
    # 5. 内部权重分配
    target_weights = {}
    
    # 股票部分
    if adjusted_stock_weight > 0:
        if broad_etf:
            # 宽基ETF占股票部分的60%
            broad_weight = adjusted_stock_weight * 0.6
            target_weights[broad_etf] = broad_weight
        
        # 主题ETF占股票部分的40%
        if theme_etfs:
            # 计算每个主题ETF的权重
            theme_weight_per_etf = (adjusted_stock_weight * 0.4) / len(theme_etfs)
            for theme_etf in theme_etfs:
                # 应用拥挤度调整
                crowded_factor = crowded_adjustments.get(theme_etf, 1.0)
                weight = theme_weight_per_etf * crowded_factor
                target_weights[theme_etf] = weight
        else:
            # 若没有主题ETF，将所有股票权重分配给宽基ETF
            if broad_etf in target_weights:
                target_weights[broad_etf] = adjusted_stock_weight
            else:
                target_weights[broad_etf] = adjusted_stock_weight
    
    # 债券部分
    if bond_available and adjusted_bond_weight > 0:
        target_weights[bond_etf] = adjusted_bond_weight
    
    # 商品部分
    if commodity_available and adjusted_commodity_weight > 0:
        # 等权分配商品ETF
        for etf in commodity_etfs:
            if etf in liquid_etfs:
                target_weights[etf] = commodity_weight_per_etf
            else:
                print(f"Warning: Commodity ETF {etf} not in liquid ETFs, skipping")
    
    # 6. 归一化与清理
    # 移除权重 ≤ 0 的ETF
    target_weights = {k: v for k, v in target_weights.items() if v > 0}
    
    # 归一化确保总和为1.0
    total_weight = sum(target_weights.values())
    if total_weight > 0:
        target_weights = {k: v / total_weight for k, v in target_weights.items()}
    
    # 7. 构建返回结果
    # 找出拥挤的ETF
    crowded_etfs = [code for code, factor in crowded_adjustments.items() if factor < 1.0]
    
    result = {
        "target_weights": target_weights,
        "reasons": {
            "macro_regime": regime,
            "equity_score": equity_friendly_score,
            "all_themes": [theme['theme'] for theme in current_themes] if current_themes else [],
            "crowded_etfs": crowded_etfs
        }
    }
    
    return result

if __name__ == "__main__":
    # 测试函数
    mock_macro = {"regime": "recovery", "equity_friendly_score": 0.68}
    mock_market = {
        "liquid_etfs": ["510300", "515980", "511260", "518880"],
        "crowded_adjustments": {"515980": 0.75}
    }
    
    try:
        # 测试使用人工智能主题
        result = build_portfolio(mock_macro, mock_market, current_themes=[{"theme": "人工智能"}])
        print("Target weights (with AI theme):")
        for code, weight in result["target_weights"].items():
            print(f"  {code}: {weight:.4f}")
        print("\nReasons:")
        for key, value in result["reasons"].items():
            print(f"  {key}: {value}")
        
        # 测试不使用主题
        result_no_theme = build_portfolio(mock_macro, mock_market, current_themes=None)
        print("\nTarget weights (no theme):")
        for code, weight in result_no_theme["target_weights"].items():
            print(f"  {code}: {weight:.4f}")
    except Exception as e:
        print(f"Error: {e}")
