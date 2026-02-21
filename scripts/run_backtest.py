#!/usr/bin/env python3
"""使用GeneralBacktest对position.csv进行回测并生成可视化报告"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from GeneralBacktest import GeneralBacktest

def load_etf_data(data_path):
    """加载ETF价格数据"""
    print(f"加载ETF价格数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期和代码排序
    df = df.sort_values(['date', 'code'])
    
    return df

def get_trading_days(etf_data):
    """从ETF价格数据中获取交易日列表"""
    trading_days = sorted(etf_data['date'].unique())
    print(f"获取到 {len(trading_days)} 个交易日")
    print(f"交易日范围: {trading_days[0]} 至 {trading_days[-1]}")
    return trading_days

def find_next_trading_day(target_date, trading_days):
    """找到距离指定日期最近的未来交易日"""
    target_date = pd.to_datetime(target_date)
    
    # 遍历交易日列表，找到第一个大于等于目标日期的交易日
    for trading_day in trading_days:
        if trading_day >= target_date:
            return trading_day
    
    # 如果没有找到（目标日期在所有交易日之后），返回最后一个交易日
    return trading_days[-1]

def adjust_position_dates(position_data, trading_days):
    """调整position_data中的日期为最近的未来交易日"""
    print("\n调整position_data中的日期为最近的未来交易日...")
    
    # 转换日期格式
    position_data['date'] = pd.to_datetime(position_data['date'])
    
    # 记录原始日期和调整后的日期
    original_dates = position_data['date'].unique()
    adjusted_dates = []
    
    # 创建日期映射字典
    date_mapping = {}
    for date in original_dates:
        next_trading_day = find_next_trading_day(date, trading_days)
        date_mapping[date] = next_trading_day
        adjusted_dates.append(next_trading_day)
        if next_trading_day != date:
            print(f"  {date.date()} → {next_trading_day.date()} (非交易日)")
    
    # 应用日期映射
    position_data['adjusted_date'] = position_data['date'].map(date_mapping)
    
    # 去除重复的日期-资产组合（如果有）
    position_data = position_data.drop_duplicates(subset=['adjusted_date', 'code'], keep='last')
    
    # 将adjusted_date重命名为date
    position_data = position_data.drop('date', axis=1).rename(columns={'adjusted_date': 'date'})
    
    # 按日期和代码排序
    position_data = position_data.sort_values(['date', 'code'])
    
    print(f"调整完成，调整后的日期数量: {position_data['date'].nunique()}")
    print(f"调整后的日期范围: {position_data['date'].min().date()} 至 {position_data['date'].max().date()}")
    
    return position_data

def load_position_data(position_path):
    """加载持仓数据"""
    print(f"加载持仓数据: {position_path}")
    df = pd.read_csv(position_path)
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    return df

def main():
    print("=" * 60)
    print("使用GeneralBacktest对position.csv进行回测")
    print("=" * 60)
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据文件路径
    etf_data_path = os.path.join(root_dir, 'data', 'etf', 'etf_2025_ohlcva.csv')
    position_path = os.path.join(root_dir, 'position.csv')
    
    # 加载数据
    etf_data = load_etf_data(etf_data_path)
    position_data = load_position_data(position_path)
    
    # 获取交易日列表
    trading_days = get_trading_days(etf_data)
    
    # 调整position_data中的日期为最近的未来交易日
    position_data = adjust_position_dates(position_data, trading_days)
    
    # 转换日期格式为字符串，以便后续筛选
    position_data['date'] = position_data['date'].dt.strftime('%Y-%m-%d')
    
    # 筛选2025年的数据
    position_data_2025 = position_data[position_data['date'].str.startswith('2025-')]
    
    print(f"\n2025年持仓数据形状: {position_data_2025.shape}")
    
    # 创建GeneralBacktest实例
    print("\n创建GeneralBacktest实例...")
    bt = GeneralBacktest(
        start_date='2025-01-31',
        end_date='2025-12-31'
    )
    
    # 创建基准权重数据
    print("\n创建基准权重数据...")
    # 获取策略的所有日期（已调整为交易日）
    strategy_dates = position_data_2025['date'].unique()
    
    # 只在第一个交易日设置基准权重，后面的交易日不需要设置
    first_trading_day = strategy_dates[0]
    
    # 全天候策略基准：股债商固定比例组合
    # 股票40% + 债券40% + 商品20%
    benchmark_etfs = [
        {'code': 159925, 'name': '沪深300ETF', 'weight': 0.40},  # 股票
        {'code': 511010, 'name': '国债ETF', 'weight': 0.40},       # 债券
        {'code': 159934, 'name': '黄金ETF', 'weight': 0.20}        # 商品
    ]
    
    # 创建基准权重数据框
    benchmark_weights = pd.DataFrame({
        'date': [first_trading_day] * len(benchmark_etfs),
        'code': [etf['code'] for etf in benchmark_etfs],
        'weight': [etf['weight'] for etf in benchmark_etfs]
    })
    
    # 确保基准ETF的日期格式与策略一致
    benchmark_weights['date'] = pd.to_datetime(benchmark_weights['date']).dt.strftime('%Y-%m-%d')
    
    print(f"基准组合: 全天候策略 (股债商固定比例)")
    print(f"基准配置:")
    for etf in benchmark_etfs:
        print(f"  {etf['code']} ({etf['name']}): {etf['weight']:.2f}")
    print(f"基准数据设置: 仅在第一个交易日 {first_trading_day} 设置权重")
    
    # 运行回测
    print("\n开始回测...")
    results = bt.run_backtest(
        weights_data=position_data_2025,
        price_data=etf_data,
        benchmark_weights=benchmark_weights,
        benchmark_name="ALL WEATHER",
        buy_price='close',
        sell_price='close',
        adj_factor_col='adj_factor',
        close_price_col='close',
        date_col='date',
        asset_col='code',
        weight_col='weight',
        initial_capital=1000000,  # 初始资金100万
        transaction_cost=[0.0003, 0.0003],  # 交易成本
        slippage=0.0001  # 滑点
    )
    
    # 打印回测结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    # 回测结果分析
    print("\n回测结果分析:")
    print(f"累计收益率: {results['metrics']['累计收益率']:.2%}")
    print(f"年化收益率: {results['metrics']['年化收益率']:.2%}")
    print(f"年化波动率: {results['metrics']['年化波动率']:.2%}")
    print(f"夏普比率: {results['metrics']['夏普比率']:.2f}")
    print(f"最大回撤: {results['metrics']['最大回撤']:.2%}")
    print(f"胜率: {results['metrics']['胜率']:.2%}")
    print(f"平均换手率: {results['metrics']['平均换手率']:.2%}")
    
    # 创建results/backtest_results目录用于保存回测结果
    output_dir = os.path.join(root_dir, 'results', 'backtest_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\n创建回测结果输出目录: {output_dir}")
    
    # 保存净值曲线数据
    nav_path = os.path.join(output_dir, 'nav_series.csv')
    results['nav_series'].to_csv(nav_path)
    print(f"净值曲线数据保存到: {nav_path}")
    
    # 保存持仓数据
    positions_path = os.path.join(output_dir, 'positions.csv')
    results['positions_df'].to_csv(positions_path, index=False)
    print(f"持仓数据保存到: {positions_path}")
    
    # 保存交易记录
    trade_records_path = os.path.join(output_dir, 'trade_records.csv')
    results['trade_records'].to_csv(trade_records_path, index=False)
    print(f"交易记录保存到: {trade_records_path}")
    
    # 保存回测指标
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("回测指标\n")
        f.write("=" * 60 + "\n")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"回测指标保存到: {metrics_path}")
    
    # 生成可视化报告
    print("\n生成可视化报告...")
    
    # 创建图表输出目录
    charts_dir = os.path.join(output_dir, 'charts')
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    print(f"创建图表输出目录: {charts_dir}")
    
    # 设置matplotlib为非交互模式，避免显示图表
    import matplotlib
    matplotlib.use('Agg')
    
    # 绘制净值曲线
    print("绘制净值曲线...")
    plt.figure(figsize=(12, 6))
    results['nav_series'].plot(title='策略净值曲线', xlabel='日期', ylabel='净值')
    plt.grid(True, alpha=0.3)
    nav_chart_path = os.path.join(charts_dir, 'nav_curve.png')
    plt.savefig(nav_chart_path, bbox_inches='tight')
    plt.close()
    print(f"净值曲线保存到: {nav_chart_path}")
    
    # 绘制月度收益率
    print("绘制月度收益率...")
    monthly_returns = results['nav_series'].resample('ME').last().pct_change()
    plt.figure(figsize=(12, 6))
    monthly_returns.plot(kind='bar', title='月度收益率', xlabel='月份', ylabel='收益率')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    monthly_chart_path = os.path.join(charts_dir, 'monthly_returns.png')
    plt.savefig(monthly_chart_path, bbox_inches='tight')
    plt.close()
    print(f"月度收益率保存到: {monthly_chart_path}")
    
    # 绘制持仓权重变化
    print("绘制持仓权重变化...")
    positions_df = results['positions_df']
    pivot_positions = positions_df.pivot(index='date', columns='asset', values='weight')
    plt.figure(figsize=(12, 6))
    pivot_positions.plot(kind='area', title='持仓权重变化', xlabel='日期', ylabel='权重', stacked=True)
    plt.grid(True, alpha=0.3)
    plt.legend(title='ETF代码')
    positions_chart_path = os.path.join(charts_dir, 'positions.png')
    plt.savefig(positions_chart_path, bbox_inches='tight')
    plt.close()
    print(f"持仓权重变化保存到: {positions_chart_path}")
    
    # 尝试生成综合Dashboard
    print("生成综合Dashboard...")
    try:
        dashboard_path = os.path.join(charts_dir, 'dashboard.png')
        # 保存图表但不显示
        bt.plot_all(save_path=dashboard_path)
        print(f"综合Dashboard保存到: {dashboard_path}")
    except Exception as e:
        print(f"生成综合Dashboard失败: {e}")
        print("跳过综合Dashboard生成")
    
    print("\n" + "=" * 60)
    print("回测完成！所有结果和图表已保存到 results/backtest_results/ 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()
