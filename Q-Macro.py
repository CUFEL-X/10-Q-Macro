#!/usr/bin/env python3
"""
Q-Macro 主入口文件
整合策略生成、持仓转换和回测功能
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_step(step_name: str, script_path: str, description: str):
    """
    执行单个步骤
    
    Args:
        step_name: 步骤名称
        script_path: 脚本路径
        description: 步骤描述
    
    Returns:
        bool: 是否成功
    """
    print("\n" + "=" * 70)
    print(f"步骤 {step_name}: {description}")
    print("=" * 70)
    
    try:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.abspath(__file__))
        full_script_path = os.path.join(root_dir, script_path)
        
        # 执行脚本
        result = subprocess.run(
            [sys.executable, full_script_path],
            cwd=root_dir,
            check=True,
            capture_output=False
        )
        
        print(f"\n✅ 步骤 {step_name} 完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 步骤 {step_name} 失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 步骤 {step_name} 出错: {e}")
        return False

def main():
    """
    主函数：执行完整的Q-Macro流程
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Q-Macro 宏观驱动ETF配置策略系统")
    parser.add_argument('--skip-step1', action='store_true', help='跳过步骤1（批量生成策略）')
    parser.add_argument('--skip-step2', action='store_true', help='跳过步骤2（持仓转换）')
    parser.add_argument('--skip-step3', action='store_true', help='跳过步骤3（回测）')
    parser.add_argument('--only-backtest', action='store_true', help='仅运行回测（跳过步骤1和2）')
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("Q-Macro 宏观驱动ETF配置策略系统")
    print("=" * 70)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 定义执行步骤
    all_steps = [
        {
            "name": "1",
            "script": "scripts/run_all_pipline.py",
            "description": "批量生成2025年全年投资策略",
            "skip": args.skip_step1 or args.only_backtest
        },
        {
            "name": "2",
            "script": "scripts/parse_portfolios_to_csv.py",
            "description": "将持仓JSON转换为CSV格式",
            "skip": args.skip_step2 or args.only_backtest
        },
        {
            "name": "3",
            "script": "scripts/run_backtest.py",
            "description": "运行回测并生成可视化报告",
            "skip": args.skip_step3
        }
    ]
    
    # 记录执行结果
    results = []
    
    # 依次执行每个步骤
    for step in all_steps:
        if step["skip"]:
            print(f"\n⏭️  跳过步骤 {step['name']}: {step['description']}")
            results.append({
                "step": step["name"],
                "description": step["description"],
                "success": True,
                "skipped": True
            })
            continue
        
        success = run_step(
            step_name=step["name"],
            script_path=step["script"],
            description=step["description"]
        )
        results.append({
            "step": step["name"],
            "description": step["description"],
            "success": success,
            "skipped": False
        })
        
        # 如果某个步骤失败，询问是否继续
        if not success:
            print(f"\n⚠️  步骤 {step['name']} 执行失败")
            print("是否继续执行后续步骤？(y/n): ", end="")
            try:
                choice = input().strip().lower()
                if choice != 'y':
                    print("用户选择终止执行")
                    break
            except:
                print("无法获取用户输入，终止执行")
                break
    
    # 计算总耗时
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 打印执行摘要
    print("\n" + "=" * 70)
    print("执行摘要")
    print("=" * 70)
    
    for result in results:
        if result["skipped"]:
            status = "⏭️  跳过"
        elif result["success"]:
            status = "✅ 成功"
        else:
            status = "❌ 失败"
        print(f"步骤 {result['step']}: {result['description']} - {status}")
    
    print("\n" + "-" * 70)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("=" * 70)
    
    # 检查是否所有步骤都成功
    all_success = all(result["success"] for result in results)
    
    if all_success:
        print("\n🎉 所有步骤执行成功！")
        print("\n📁 输出文件位置:")
        print("  - 投资组合结果: portfolios/portfolio_result_YYYY-MM-DD.json")
        print("  - 持仓CSV文件: position.csv")
        print("  - 回测结果: results/backtest_results/")
        print("  - 投资策略报告: reports/investment_report_YYYY-MM-DD.md")
    else:
        print("\n⚠️  部分步骤执行失败，请检查错误信息")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
