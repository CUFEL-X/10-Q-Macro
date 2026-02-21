"""解析持仓JSON文件并保存为CSV格式"""
import json
import os
import pandas as pd
from datetime import datetime

def parse_portfolio_json_to_csv(
    portfolios_dir: str = "portfolios",
    output_file: str = "position.csv"
):
    """
    解析portfolios文件夹内的持仓JSON文件，并保存为CSV格式
    
    Args:
        portfolios_dir: portfolios文件夹路径
        output_file: 输出CSV文件路径
    """
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 构建portfolios文件夹路径
    portfolios_path = os.path.join(project_root, portfolios_dir)
    
    # 检查文件夹是否存在
    if not os.path.exists(portfolios_path):
        print(f"Error: Portfolios directory not found: {portfolios_path}")
        return
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(portfolios_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {portfolios_path}")
        return
    
    print(f"Found {len(json_files)} JSON files in {portfolios_path}")
    
    # 解析所有JSON文件
    all_positions = []
    
    for json_file in sorted(json_files):
        # 从文件名中提取日期
        # 文件名格式: portfolio_result_YYYY-MM-DD.json
        date_str = json_file.replace('portfolio_result_', '').replace('.json', '')
        
        # 验证日期格式
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print(f"Warning: Invalid date format in filename: {json_file}, skipping...")
            continue
        
        # 读取JSON文件
        json_path = os.path.join(portfolios_path, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}, skipping...")
            continue
        
        # 提取target_weights
        target_weights = data.get('target_weights', {})
        
        if not target_weights:
            print(f"Warning: No target_weights found in {json_file}, skipping...")
            continue
        
        # 将权重数据添加到列表
        for code, weight in target_weights.items():
            all_positions.append({
                'date': date,
                'code': code,
                'weight': round(weight, 4)  # 保留4位小数
            })
    
    # 创建DataFrame
    df = pd.DataFrame(all_positions)
    
    # 按日期和代码排序
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # 保存为CSV
    output_path = os.path.join(project_root, output_file)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nSuccessfully saved {len(df)} positions to {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Unique ETFs: {df['code'].nunique()}")
    
    # 显示前几行
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    
    return df

if __name__ == "__main__":
    # 解析持仓JSON文件并保存为CSV
    df = parse_portfolio_json_to_csv()
