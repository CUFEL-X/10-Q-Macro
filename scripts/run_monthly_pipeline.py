"""月度策略执行主流程"""
import argparse
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.macro_regime import detect_macro_regime_and_score
from src.core.market_calibration import calibrate_etf_market_conditions
from src.agents.policy_interpreter import interpret_policy
from src.core.portfolio_builder import build_portfolio
from src.agents.report_writer import generate_monthly_report, MacroState, PolicySignal, MarketCondition, Portfolio

def run_monthly(target_date: str) -> dict:
    """
    执行月度策略流程。
    
    Args:
        target_date: 目标日期，格式为 "YYYY-MM-DD"
    
    Returns:
        dict: 包含投资组合权重和决策原因
    """
    print(f"=== 开始执行月度策略流程 (目标日期: {target_date}) ===")
    
    # 1. 宏观状态分析
    print("\n1. 执行宏观状态分析...")
    try:
        macro = detect_macro_regime_and_score(target_date)
        print(f"   宏观状态: {macro['regime']}")
        print(f"   权益友好度评分: {macro['equity_friendly_score']:.4f}")
    except Exception as e:
        print(f"   错误: {e}")
        raise
    
    # 2. 市场校准
    print("\n2. 执行市场校准...")
    try:
        market = calibrate_etf_market_conditions(target_date)
        liquid_count = len(market.get('liquid_etfs', []))
        print(f"   流动性合格ETF数量: {liquid_count}")
        
        crowded_count = len([k for k, v in market.get('crowded_adjustments', {}).items() if v < 1.0])
        print(f"   拥挤ETF数量: {crowded_count}")
    except Exception as e:
        print(f"   错误: {e}")
        raise
    
    # 3. 主题决策（使用政策解读）
    print("\n3. 执行主题决策...")
    try:
        policy_signal = interpret_policy(
            policy_path="data/policy_texts/govcn_2025.csv",
            target_date=target_date,
            macro_context=macro
        )
        
        # 打印所有5个主题的分析
        print("   所有推荐主题分析:")
        if policy_signal.get("top_5_themes"):
            for i, theme_info in enumerate(policy_signal["top_5_themes"], 1):
                print(f"   {i}. 主题: {theme_info['theme']}, 置信度: {theme_info['confidence']:.4f}")
                if theme_info.get('evidence'):
                    for j, evidence in enumerate(theme_info['evidence'], 1):
                        print(f"      理由{j}: {evidence}")
    except Exception as e:
        print(f"   错误: {e}")
        raise
    
    # 4. 构建组合
    print("\n4. 构建ETF投资组合...")
    try:
        portfolio = build_portfolio(
            macro_state=macro,
            market_cond=market,
            current_themes=policy_signal.get("top_5_themes", [])
        )
        
        # 打印组合权重
        print("   目标权重:")
        for code, weight in portfolio["target_weights"].items():
            print(f"     {code}: {weight:.4f}")
        
        # 打印决策原因
        print("\n   决策原因:")
        for key, value in portfolio["reasons"].items():
            # 跳过theme_used字段
            if key != "theme_used":
                print(f"     {key}: {value}")
        
        # 特别打印所有推荐主题
        if portfolio.get("reasons") and portfolio["reasons"].get("all_themes"):
            print(f"     all_recommended_themes: {', '.join(portfolio['reasons']['all_themes'])}")
    except Exception as e:
        print(f"   错误: {e}")
        raise
    
    # 5. 生成投资策略报告
    print("\n5. 生成投资策略报告...")
    try:
        # 构建报告所需的数据模型
        macro_state = MacroState(
            regime=macro['regime'],
            equity_friendly_score=macro['equity_friendly_score'],
            growth_momentum=macro.get('growth_momentum', 0.0),
            inflation_momentum=macro.get('inflation_momentum', 0.0)
        )
        
        # 收集所有5个主题的证据
        all_evidence = []
        if policy_signal.get('top_5_themes'):
            for theme_info in policy_signal['top_5_themes']:
                if theme_info.get('evidence'):
                    all_evidence.extend(theme_info['evidence'])
        # 去重并限制数量
        all_evidence = list(set(all_evidence))[:5]
        
        policy_signal = PolicySignal(
            use_thematic=policy_signal['use_thematic'],
            confidence=policy_signal['confidence'],
            evidence=all_evidence if all_evidence else ['政策分析支持相关主题']
        )
        
        # 确保liquid_etfs中的元素是字符串类型
        liquid_etfs = [str(etf) for etf in market.get('liquid_etfs', [])]
        
        # 确保crowded_themes的键是字符串类型
        crowded_themes = {str(k): v for k, v in market.get('crowded_adjustments', {}).items() if v < 1.0}
        
        market_cond = MarketCondition(
            liquid_etfs=liquid_etfs,
            crowded_themes=crowded_themes
        )
        
        # 转换投资组合数据格式
        stocks = []
        bonds = []
        commodities = []
        
        for code, weight in portfolio['target_weights'].items():
            # 这里简化处理，实际应根据ETF类型分类
            stocks.append({"etf_code": code, "weight": weight})
        
        portfolio_data = Portfolio(
            stocks=stocks,
            bonds=bonds,
            commodities=commodities
        )
        
        # 生成报告
        report = generate_monthly_report(
            target_date=target_date,
            macro_state=macro_state,
            policy_signal=policy_signal,
            market_cond=market_cond,
            portfolio=portfolio_data
        )
        
        # 保存报告到文件
        report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f"investment_report_{target_date}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"   报告已生成并保存至: {report_file}")
        
    except Exception as e:
        print(f"   生成报告时出错: {e}")
    
    # 6. 保存投资组合结果到JSON文件
    import json
    portfolios_dir = os.path.join(os.path.dirname(__file__), '..', 'portfolios')
    os.makedirs(portfolios_dir, exist_ok=True)
    
    portfolio_file = os.path.join(portfolios_dir, f"portfolio_result_{target_date}.json")
    with open(portfolio_file, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)
    print(f"\n   投资组合结果已保存至: {portfolio_file}")
    
    print("\n=== 月度策略流程执行完成 ===")
    return portfolio

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="执行月度ETF策略流程")
    parser.add_argument(
        "--date", 
        type=str, 
        default=datetime.now().strftime("%Y-%m-%d"),
        help="目标日期，格式为 YYYY-MM-DD (默认: 今天)"
    )
    
    args = parser.parse_args()
    
    try:
        # 验证日期格式
        datetime.strptime(args.date, "%Y-%m-%d")
        
        # 执行策略
        result = run_monthly(args.date)
        
    except ValueError as e:
        print(f"日期格式错误: {e}")
        print("请使用 YYYY-MM-DD 格式，例如: 2025-12-31")
    except Exception as e:
        print(f"执行错误: {e}")

if __name__ == "__main__":
    main()
