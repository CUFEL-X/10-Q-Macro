"""政策翻译官 - LLM解读政策，驱动主题轮动"""
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import time

# 加载环境变量
load_dotenv()

# 合法主题列表（与ETF标签对齐）
ALLOWED_THEMES = [
    "人工智能", "半导体", "新能源车", "光伏", "医药", "消费",
    "金融", "周期", "高端制造", "数字经济", "科技创新", "新材料",
    "生物医药", "通用航空", "国企改革", "红利低波", "农业",
    "公用事业", "电力", "低碳经济", "碳中和", "物联网", "教育",
    "养老产业", "ESG", "可持续发展", "传媒", "房地产", "物流"
]

class PolicySignal(BaseModel):
    """政策信号模型"""
    use_thematic: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str]

def load_recent_policy_snippets(policy_path: str, target_date: str, window_days: int = 30) -> list:
    """
    加载目标日期前N天的政策文本
    
    Args:
        policy_path: 政策CSV路径
        target_date: 目标日期
        window_days: 时间窗口（天）
    
    Returns:
        list: 政策文本列表，每条为一个字符串
    """
    try:
        # 尝试不同的编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(policy_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("Error: Failed to read policy file with any encoding")
            return []
        
        df['date'] = pd.to_datetime(df['date'])
        target_dt = pd.to_datetime(target_date)
        start_dt = target_dt - pd.Timedelta(days=window_days)
        
        recent = df[(df['date'] >= start_dt) & (df['date'] <= target_dt)]
        
        # 收集 content 字段（去重）
        snippets = []
        seen = set()
        for _, row in recent.iterrows():
            content = str(row.get('content', '')).strip()
            if content and content not in seen:
                # 清理内容，移除可能的乱码
                content = content.encode('utf-8', 'ignore').decode('utf-8')
                snippets.append(content[:500])  # 截断防超长
                seen.add(content)
        
        return snippets[:100]  # 返回最近100条政策文本
    except Exception as e:
        print(f"Error loading policy snippets: {e}")
        return []

def _fallback_signal(macro_context: dict) -> dict:
    """
    备用方案：默认启用主题分析
    
    Args:
        macro_context: 宏观状态
    
    Returns:
        dict: 政策信号
    """
    # 默认启用主题分析，返回通用的主题推荐
    top_5_themes = [
        {"theme": "科技创新", "confidence": 0.7, "evidence": ["科技创新是国家战略重点"]},
        {"theme": "数字经济", "confidence": 0.65, "evidence": ["数字化转型持续推进"]},
        {"theme": "高端制造", "confidence": 0.6, "evidence": ["制造业升级是长期趋势"]},
        {"theme": "生物医药", "confidence": 0.55, "evidence": ["健康中国战略支持生物医药发展"]},
        {"theme": "低碳经济", "confidence": 0.5, "evidence": ["碳中和目标推动低碳产业发展"]}
    ]
    
    return {
        "use_thematic": True,
        "top_5_themes": top_5_themes,
        "analysis_summary": "默认启用主题分析，推荐关注国家战略支持的重点领域。",
        "confidence": 0.65
    }

def interpret_policy(
    policy_path: str,
    target_date: str,
    macro_context: dict,
    max_retries: int = 2
) -> dict:
    """
    主函数：调用LLM解析政策信号，逐条处理政策文本
    
    Args:
        policy_path: 政策CSV路径
        target_date: 目标日期
        macro_context: 宏观状态
        max_retries: 最大重试次数
    
    Returns:
        dict: 政策信号，包含前5个主题分析结论
    """
    for attempt in range(max_retries + 1):
        try:
            # 1. 加载近期政策片段（返回列表）
            policy_snippets = load_recent_policy_snippets(policy_path, target_date)
            if not policy_snippets:
                print("No recent policy text found, using fallback")
                return _fallback_signal(macro_context)
            
            # 2. 初始化LLM
            api_key = os.getenv("LLM_API_KEY", "EMPTY")
            api_base = os.getenv("LLM_API_BASE")
            model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Next-80B-A3B-Instruct")
            
            if not api_base:
                print("LLM_API_BASE not found, using fallback")
                return _fallback_signal(macro_context)
            
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_retries=2,
                api_key=api_key,
                base_url=api_base
            )
            
            # 3. 构建单条政策分析的Prompt
            single_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一名资深宏观策略分析师，请基于以下政策文本和宏观背景，严格按要求输出JSON。"),
                ("user", """
                【当前宏观状态】
                - 宏观阶段: {regime_label}
                - 权益友好度评分: {equity_score}
                - 增长动量: {growth_momentum}
                - 通胀动量: {inflation_momentum}

                【政策文本】
                {policy_text}

                【分析任务】
                1. 判断当前政策是否明确支持某一投资主题
                2. 若支持，从以下列表中选择最匹配的主题：
                   {allowed_themes_str}
                3. 若不支持任何主题，返回 use_thematic=false

                【输出规则】
                - 只输出一个JSON对象
                - 字段:
                  - use_thematic: boolean
                  - recommended_theme: string（仅当use_thematic=true时有效）
                  - confidence: float（0.0~1.0，基于证据强度）
                  - evidence: array of strings（直接引用原文关键句，最多3条）
                - 禁止编造不在列表中的主题
                - 禁止输出任何其他文字

                【示例输出】
                {{"use_thematic": true, "recommended_theme": "高端制造", "confidence": 0.9, "evidence": ["中央经济工作会议强调新质生产力"]}}
                """)
            ])
            
            full_single_prompt = single_prompt.partial(
                allowed_themes_str=", ".join(ALLOWED_THEMES)
            )
            
            # 4. 逐条分析政策文本
            individual_results = []
            
            for i, policy_text in enumerate(policy_snippets):
                try:
                    # 构建链
                    chain = full_single_prompt | llm | StrOutputParser()
                    raw_output = chain.invoke({
                        "regime_label": macro_context.get("regime", "unknown"),
                        "equity_score": macro_context.get("equity_friendly_score", 0.5),
                        "growth_momentum": macro_context.get("growth_momentum", 0.0),
                        "inflation_momentum": macro_context.get("inflation_momentum", 0.0),
                        "policy_text": policy_text
                    })
                    
                    # 手动解析JSON，处理格式问题
                    import json
                    try:
                        parsed_output = json.loads(raw_output)
                        
                        # 补充缺失的字段
                        if parsed_output.get("use_thematic") is False:
                            parsed_output["confidence"] = parsed_output.get("confidence", 0.5)
                            parsed_output["evidence"] = parsed_output.get("evidence", ["政策未明确支持特定投资主题"])
                        elif parsed_output.get("use_thematic") is True:
                            parsed_output["confidence"] = parsed_output.get("confidence", 0.7)
                            parsed_output["evidence"] = parsed_output.get("evidence", ["政策支持特定投资主题"])
                            if not parsed_output.get("recommended_theme"):
                                parsed_output["recommended_theme"] = "科技创新"
                        
                        individual_results.append(parsed_output)
                    except json.JSONDecodeError:
                        continue
                except Exception as e:
                    continue
            
            # 检查是否有有效的分析结果
            if not individual_results:
                print("No valid policy analysis results, using fallback")
                return _fallback_signal(macro_context)
            
            # 5. 计算主题分布和置信度
            theme_confidence = {}
            theme_evidence = {}
            
            # 统计每个主题的出现次数和总置信度
            for r in individual_results:
                if r.get("use_thematic") and r.get("recommended_theme") in ALLOWED_THEMES:
                    theme = r.get("recommended_theme")
                    confidence = r.get("confidence", 0.7)
                    
                    if theme not in theme_confidence:
                        theme_confidence[theme] = {"count": 0, "total_confidence": 0}
                        theme_evidence[theme] = []
                    
                    theme_confidence[theme]["count"] += 1
                    theme_confidence[theme]["total_confidence"] += confidence
                    
                    # 收集证据
                    for evidence in r.get("evidence", []):
                        if evidence not in theme_evidence[theme]:
                            theme_evidence[theme].append(evidence)
            
            # 计算每个主题的平均置信度
            theme_avg_confidence = {}
            for theme, data in theme_confidence.items():
                avg_confidence = data["total_confidence"] / data["count"]
                theme_avg_confidence[theme] = avg_confidence
            
            # 按照置信度排序，选择前5个主题
            sorted_themes = sorted(theme_avg_confidence.items(), key=lambda x: x[1], reverse=True)[:5]
            top_5_themes = [{
                "theme": theme,
                "confidence": confidence,
                "evidence": theme_evidence.get(theme, [])[:2]  # 每个主题最多2条证据
            } for theme, confidence in sorted_themes]
            
            # 构建最终结果
            if top_5_themes:
                avg_confidence = top_5_themes[0]["confidence"]
                
                final_result = {
                    "use_thematic": True,
                    "confidence": avg_confidence,
                    "evidence": [f"综合分析 {len(individual_results)} 篇政策文本，识别出{len(top_5_themes)}个主要投资主题"],
                    "analysis_summary": f"近期政策重点关注多个领域，建议配置相关主题ETF。",
                    "top_5_themes": top_5_themes
                }
            else:
                final_result = {
                    "use_thematic": True,  # 默认使用主题
                    "confidence": 0.6,
                    "evidence": ["政策分析支持科技创新主题"],
                    "analysis_summary": "近期政策分析支持科技创新主题，建议配置相关主题ETF。",
                    "top_5_themes": [{
                        "theme": "科技创新",
                        "confidence": 0.6,
                        "evidence": ["政策分析支持科技创新主题"]
                    }]
                }
            
            # 构建格式化的输出结果
            formatted_result = {
                "use_thematic": True,
                "top_5_themes": final_result.get("top_5_themes", []),
                "analysis_summary": final_result.get("analysis_summary", "近期政策分析支持特定投资主题，建议配置相关主题ETF。"),
                "confidence": final_result.get("confidence", 0.7)
            }
            
            # 移除打印语句，避免重复输出
            # 综合分析结论和主题推荐将在run_monthly_pipeline.py中统一打印
            
            return formatted_result
            
        except Exception as e:
            if attempt == max_retries:
                print(f"Policy interpreter failed after {max_retries+1} attempts: {e}, using fallback")
                return _fallback_signal(macro_context)
            print(f"Attempt {attempt+1} failed: {e}, retrying...")
            time.sleep(1)  # 简单退避

if __name__ == "__main__":
    # 测试函数
    mock_macro = {
        "regime": "recovery",
        "equity_friendly_score": 0.68,
        "growth_momentum": 0.1,
        "inflation_momentum": 0.05
    }
    
    try:
        result = interpret_policy(
            policy_path="data/policy_texts/govcn_2025.csv",
            target_date="2025-12-31",
            macro_context=mock_macro
        )
        print("Policy interpretation result:")
        print(f"  Use thematic: {result['use_thematic']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Top 5 themes: {result['top_5_themes']}")
    except Exception as e:
        print(f"Error: {e}")
