"""批量执行全年策略流程"""
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_monthly_pipeline import run_monthly

def generate_monthly_dates(year: int) -> list:
    """
    生成指定年份每个月的最后一天日期
    
    Args:
        year: 年份
    
    Returns:
        list: 日期字符串列表，格式为YYYY-MM-DD
    """
    dates = []
    for month in range(1, 13):
        # 计算下个月的第一天，然后减去一天得到当月最后一天
        if month == 12:
            # 12月的下一个月是次年1月
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        last_day = next_month - timedelta(days=1)
        dates.append(last_day.strftime("%Y-%m-%d"))
    
    return dates

def run_all(year: int = 2025):
    """
    执行指定年份所有月份的策略流程
    
    Args:
        year: 年份，默认为2025
    """
    print(f"=== 开始执行{year}年全年策略流程 ===")
    
    # 生成所有月份的日期
    monthly_dates = generate_monthly_dates(year)
    
    # 记录成功和失败的月份
    success_months = []
    failed_months = []
    
    # 循环执行每个月的策略
    for date in monthly_dates:
        print(f"\n=== 执行{date}策略流程 ===")
        try:
            # 执行月度策略
            result = run_monthly(date)
            success_months.append(date)
            print(f"✅ {date}策略执行成功")
        except Exception as e:
            print(f"❌ {date}策略执行失败: {e}")
            failed_months.append(date)
        
        # 暂停一下，避免请求过快
        import time
        time.sleep(2)
    
    # 打印执行结果汇总
    print(f"\n=== {year}年全年策略流程执行完成 ===")
    print(f"成功月份: {len(success_months)}/{len(monthly_dates)}")
    if success_months:
        print(f"  成功列表: {', '.join(success_months)}")
    
    if failed_months:
        print(f"失败月份: {len(failed_months)}/{len(monthly_dates)}")
        print(f"  失败列表: {', '.join(failed_months)}")

if __name__ == "__main__":
    # 默认执行2025年全年策略
    run_all(2025)
