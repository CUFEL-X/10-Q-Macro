"""归因分析师"""
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import time


class MacroState(BaseModel):
    """宏观状态数据模型"""
    regime: str  # 当前经济周期（复苏、过热、滞胀、衰退）
    equity_friendly_score: float  # 权益友好度评分
    growth_momentum: float  # 增长动量
    inflation_momentum: float  # 通胀动量


class PolicySignal(BaseModel):
    """政策信号数据模型"""
    use_thematic: bool  # 是否使用主题投资
    confidence: float  # 置信度
    evidence: List[str]  # 政策支持的证据


class MarketCondition(BaseModel):
    """市场条件数据模型"""
    liquid_etfs: List[str]  # 流动性良好的ETF
    crowded_themes: Dict[str, float]  # 拥挤度分析


class Portfolio(BaseModel):
    """投资组合数据模型"""
    stocks: List[Dict[str, Any]]  # 股票ETF权重
    bonds: List[Dict[str, Any]]  # 债券ETF权重
    commodities: List[Dict[str, Any]]  # 商品ETF权重


# 加载环境变量
load_dotenv()

# 初始化LLM
def init_llm():
    """
    初始化LLM模型
    
    Returns:
        ChatOpenAI: 初始化后的LLM模型
    """
    api_key = os.getenv("LLM_API_KEY", "EMPTY")
    api_base = os.getenv("LLM_API_BASE")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Next-80B-A3B-Instruct")
    
    if not api_base:
        print("Warning: LLM_API_BASE not found, using fallback report generation")
        return None
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # 低温度，确保输出稳定
            max_retries=2,
            api_key=api_key,
            base_url=api_base
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None


# 初始化LLM实例
llm = init_llm()


def generate_detailed_macro_description(macro_state: MacroState) -> str:
    """
    使用LLM生成详细的宏观状态描述
    
    Args:
        macro_state: 宏观状态数据
    
    Returns:
        详细的宏观状态描述
    """
    global llm
    
    if not llm:
        # 回退方案：使用原始的简单描述
        return describe_macro_state(macro_state)
    
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名资深宏观策略分析师，请基于以下宏观经济数据，生成详细的宏观状态分析。"),
        ("user", """
        【宏观经济数据】
        - 当前经济周期: {regime}
        - 权益友好度评分: {equity_score}
        - 增长动量: {growth_momentum}
        - 通胀动量: {inflation_momentum}

        【分析要求】
        1. 详细分析当前经济所处的阶段特征
        2. 解释权益友好度评分的含义及其对股市的影响
        3. 分析增长动量和通胀动量的组合对投资策略的启示
        4. 基于以上分析，提出对股票、债券、商品等大类资产的配置建议
        5. 分析应专业、客观、详细，不少于200字

        【输出格式】
        - 使用中文
        - 分段落输出，每段有明确的主题
        - 避免使用过于技术化的术语，确保可读性
        """)
    ])
    
    # 构建链
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 调用LLM生成分析
        description = chain.invoke({
            "regime": macro_state.regime,
            "equity_score": macro_state.equity_friendly_score,
            "growth_momentum": macro_state.growth_momentum,
            "inflation_momentum": macro_state.inflation_momentum
        })
        
        return description
    except Exception as e:
        print(f"Error generating macro description: {e}")
        # 回退方案
        return describe_macro_state(macro_state)


def describe_macro_state(macro_state: MacroState) -> str:
    """
    描述宏观状态（回退方案）
    
    Args:
        macro_state: 宏观状态数据
    
    Returns:
        宏观状态的自然语言描述
    """
    regime = macro_state.regime
    score = macro_state.equity_friendly_score
    growth = macro_state.growth_momentum
    inflation = macro_state.inflation_momentum
    
    description = f"当前中国经济处于**{regime}阶段**（PMI=52.1），权益友好度评分为{score:.2f}，表明股市环境{'较为有利' if score > 0.6 else '一般'}。\n"
    
    return description


def generate_detailed_policy_interpretation(policy_signal: PolicySignal) -> str:
    """
    使用LLM生成详细的政策解读
    
    Args:
        policy_signal: 政策信号数据
    
    Returns:
        详细的政策解读
    """
    global llm
    
    if not llm:
        # 回退方案：使用原始的简单解读
        return interpret_policy_signal(policy_signal)
    
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名资深政策分析师，请基于以下政策信号数据，生成详细的政策解读。"),
        ("user", """
        【政策信号数据】
        - 是否使用主题投资: {use_thematic}
        - 置信度: {confidence}
        - 政策支持的证据: {evidence}

        【分析要求】
        1. 详细分析政策信号对投资主题的影响
        2. 解释政策支持对相关行业和企业的影响
        3. 分析政策信号的可持续性和潜在风险
        4. 基于政策分析，提出具体的投资策略建议
        5. 分析应专业、客观、详细，不少于200字

        【输出格式】
        - 使用中文
        - 分段落输出，每段有明确的主题
        - 避免使用过于技术化的术语，确保可读性
        """)
    ])
    
    # 构建链
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 调用LLM生成分析
        interpretation = chain.invoke({
            "use_thematic": policy_signal.use_thematic,
            "confidence": policy_signal.confidence,
            "evidence": "\n".join(policy_signal.evidence)
        })
        
        return interpretation
    except Exception as e:
        print(f"Error generating policy interpretation: {e}")
        # 回退方案
        return interpret_policy_signal(policy_signal)


def interpret_policy_signal(policy_signal: PolicySignal) -> str:
    """
    解读政策信号（回退方案）
    
    Args:
        policy_signal: 政策信号数据
    
    Returns:
        政策解读的自然语言描述
    """
    if not policy_signal.use_thematic:
        return "未发现明确的主题推荐。\n"
    
    confidence = policy_signal.confidence
    evidence = "\n".join([f"- {ev}" for ev in policy_signal.evidence])
    
    interpretation = (
        f"近期政策重点支持多个投资主题，具体体现在如下证据：\n"
        f"{evidence}\n"
        f"这一信号增强了我们对相关主题的信心（置信度={confidence:.2f}）。\n"
    )
    
    return interpretation


def generate_detailed_market_analysis(market_cond: MarketCondition) -> str:
    """
    使用LLM生成详细的市场条件分析
    
    Args:
        market_cond: 市场条件数据
    
    Returns:
        详细的市场条件分析
    """
    global llm
    
    if not llm:
        # 回退方案：使用原始的简单分析
        return analyze_market_conditions(market_cond)
    
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名资深市场分析师，请基于以下市场条件数据，生成详细的市场分析。"),
        ("user", """
        【市场条件数据】
        - 流动性良好的ETF: {liquid_etfs}
        - 拥挤度分析: {crowded_themes}

        【分析要求】
        1. 详细分析ETF市场的流动性状况及其对投资的影响
        2. 解释交易拥挤度的含义及其对投资风险的警示作用
        3. 分析流动性和拥挤度的组合对投资策略的启示
        4. 基于市场条件分析，提出具体的ETF选择和配置建议
        5. 分析应专业、客观、详细，不少于200字

        【输出格式】
        - 使用中文
        - 分段落输出，每段有明确的主题
        - 避免使用过于技术化的术语，确保可读性
        """)
    ])
    
    # 构建链
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 调用LLM生成分析
        analysis = chain.invoke({
            "liquid_etfs": ", ".join(market_cond.liquid_etfs),
            "crowded_themes": "\n".join([f"{theme}: Z-score={z:.1f}" for theme, z in market_cond.crowded_themes.items()])
        })
        
        return analysis
    except Exception as e:
        print(f"Error generating market analysis: {e}")
        # 回退方案
        return analyze_market_conditions(market_cond)


def analyze_market_conditions(market_cond: MarketCondition) -> str:
    """
    分析市场条件（回退方案）
    
    Args:
        market_cond: 市场条件数据
    
    Returns:
        市场条件分析的自然语言描述
    """
    liquid_etfs = ", ".join(market_cond.liquid_etfs)
    crowded_themes = "\n".join([f"- {theme}: Z-score={z:.1f}" for theme, z in market_cond.crowded_themes.items()])
    
    analysis = (
        f"- **流动性状况**：筛选出日均成交额超过100万的ETF包括{liquid_etfs}。\n"
        f"- **拥挤度警告**：检测到以下主题存在交易拥挤情况：\n{crowded_themes}\n"
    )
    
    return analysis


def generate_detailed_portfolio_description(portfolio: Portfolio) -> str:
    """
    使用LLM生成详细的投资组合描述
    
    Args:
        portfolio: 投资组合数据
    
    Returns:
        详细的投资组合描述
    """
    global llm
    
    if not llm:
        # 回退方案：使用原始的简单描述
        return describe_portfolio(portfolio)
    
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名资深投资组合分析师，请基于以下投资组合数据，生成详细的投资组合分析。"),
        ("user", """
        【投资组合数据】
        - 股票ETF权重: {stocks}
        - 债券ETF权重: {bonds}
        - 商品ETF权重: {commodities}

        【分析要求】
        1. 详细分析投资组合的资产配置策略和逻辑
        2. 解释各ETF的选择理由和权重分配的依据
        3. 分析投资组合的风险收益特征
        4. 基于投资组合分析，提出具体的投资建议和风险管理策略
        5. 分析应专业、客观、详细，不少于200字

        【输出格式】
        - 使用中文
        - 分段落输出，每段有明确的主题
        - 避免使用过于技术化的术语，确保可读性
        """)
    ])
    
    # 构建链
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 准备数据
        stocks_str = "\n".join([f"{etf['etf_code']}: {etf['weight'] * 100:.0f}%" for etf in portfolio.stocks])
        bonds_str = "\n".join([f"{etf['etf_code']}: {etf['weight'] * 100:.0f}%" for etf in portfolio.bonds]) if portfolio.bonds else "无"
        commodities_str = "\n".join([f"{etf['etf_code']}: {etf['weight'] * 100:.0f}%" for etf in portfolio.commodities]) if portfolio.commodities else "无"
        
        # 调用LLM生成分析
        description = chain.invoke({
            "stocks": stocks_str,
            "bonds": bonds_str,
            "commodities": commodities_str
        })
        
        return description
    except Exception as e:
        print(f"Error generating portfolio description: {e}")
        # 回退方案
        return describe_portfolio(portfolio)


def describe_portfolio(portfolio: Portfolio) -> str:
    """
    描述投资组合（回退方案）
    
    Args:
        portfolio: 投资组合数据
    
    Returns:
        投资组合的自然语言描述
    """
    stocks = "\n".join([f"- **{etf['etf_code']}**: 占比{etf['weight'] * 100:.0f}%" for etf in portfolio.stocks])
    
    description = (
        f"本次策略配置了以下ETF：\n"
        f"{stocks}\n"
    )
    
    return description


def generate_monthly_report(
    target_date: str,
    macro_state: MacroState,
    policy_signal: PolicySignal,
    market_cond: MarketCondition,
    portfolio: Portfolio
) -> str:
    """
    生成月度投资策略报告
    
    Args:
        target_date: 目标日期
        macro_state: 宏观状态数据
        policy_signal: 政策信号数据
        market_cond: 市场条件数据
        portfolio: 投资组合数据
    
    Returns:
        完整的投资策略报告（Markdown格式）
    """
    # 使用LLM生成详细分析
    macro_description = generate_detailed_macro_description(macro_state)
    policy_interpretation = generate_detailed_policy_interpretation(policy_signal)
    market_analysis = generate_detailed_market_analysis(market_cond)
    portfolio_description = generate_detailed_portfolio_description(portfolio)
    
    # 生成完整报告
    report = (
        f"# {target_date} 投资策略报告\n\n"
        f"## 宏观状态概览\n"
        f"{macro_description}\n\n"
        f"## 政策解读\n"
        f"{policy_interpretation}\n\n"
        f"## 市场条件分析\n"
        f"{market_analysis}\n\n"
        f"## 投资组合构建\n"
        f"{portfolio_description}\n\n"
        f"## 投资决策总结\n"
        f"{generate_investment_summary(macro_state, policy_signal, market_cond, portfolio)}\n"
    )
    
    return report


def generate_investment_summary(
    macro_state: MacroState,
    policy_signal: PolicySignal,
    market_cond: MarketCondition,
    portfolio: Portfolio
) -> str:
    """
    使用LLM生成投资决策总结
    
    Args:
        macro_state: 宏观状态数据
        policy_signal: 政策信号数据
        market_cond: 市场条件数据
        portfolio: 投资组合数据
    
    Returns:
        投资决策总结
    """
    global llm
    
    if not llm:
        # 回退方案：使用简单的总结
        return "基于宏观状态、政策信号和市场条件分析，我们构建了当前的投资组合。投资决策考虑了多种因素，旨在实现风险与收益的平衡。"
    
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名资深投资策略分析师，请基于以下完整的投资分析数据，生成详细的投资决策总结。"),
        ("user", """
        【投资分析数据】
        1. 宏观状态：
           - 当前经济周期: {regime}
           - 权益友好度评分: {equity_score}
           - 增长动量: {growth_momentum}
           - 通胀动量: {inflation_momentum}
        
        2. 政策信号：
           - 是否使用主题投资: {use_thematic}
           - 置信度: {confidence}
           - 政策支持的证据: {evidence}
        
        3. 市场条件：
           - 流动性良好的ETF: {liquid_etfs}
           - 拥挤度分析: {crowded_themes}
        
        4. 投资组合：
           - 股票ETF权重: {stocks}
           - 债券ETF权重: {bonds}
           - 商品ETF权重: {commodities}

        【分析要求】
        1. 综合分析以上所有数据，总结投资决策的核心逻辑
        2. 解释各因素对最终投资组合构建的影响权重
        3. 分析当前投资策略的优势和潜在风险
        4. 提出未来可能需要调整的情景和应对策略
        5. 总结应专业、客观、详细，不少于200字

        【输出格式】
        - 使用中文
        - 分段落输出，每段有明确的主题
        - 避免使用过于技术化的术语，确保可读性
        """)
    ])
    
    # 构建链
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 准备数据
        stocks_str = "\n".join([f"{etf['etf_code']}: {etf['weight'] * 100:.0f}%" for etf in portfolio.stocks])
        bonds_str = "\n".join([f"{etf['etf_code']}: {etf['weight'] * 100:.0f}%" for etf in portfolio.bonds]) if portfolio.bonds else "无"
        commodities_str = "\n".join([f"{etf['etf_code']}: {etf['weight'] * 100:.0f}%" for etf in portfolio.commodities]) if portfolio.commodities else "无"
        
        # 调用LLM生成分析
        summary = chain.invoke({
            "regime": macro_state.regime,
            "equity_score": macro_state.equity_friendly_score,
            "growth_momentum": macro_state.growth_momentum,
            "inflation_momentum": macro_state.inflation_momentum,
            "use_thematic": policy_signal.use_thematic,
            "confidence": policy_signal.confidence,
            "evidence": "\n".join(policy_signal.evidence),
            "liquid_etfs": ", ".join(market_cond.liquid_etfs),
            "crowded_themes": "\n".join([f"{theme}: Z-score={z:.1f}" for theme, z in market_cond.crowded_themes.items()]),
            "stocks": stocks_str,
            "bonds": bonds_str,
            "commodities": commodities_str
        })
        
        return summary
    except Exception as e:
        print(f"Error generating investment summary: {e}")
        # 回退方案
        return "基于宏观状态、政策信号和市场条件分析，我们构建了当前的投资组合。投资决策考虑了多种因素，旨在实现风险与收益的平衡。"


if __name__ == "__main__":
    """测试代码"""
    # 测试数据
    test_macro_state = MacroState(
        regime="复苏",
        equity_friendly_score=0.72,
        growth_momentum=0.8,
        inflation_momentum=0.2
    )
    
    test_policy_signal = PolicySignal(
        use_thematic=True,
        confidence=0.9,
        evidence=["中国以系统化的顶层设计推动科技创新"]
    )
    
    test_market_cond = MarketCondition(
        liquid_etfs=["159208 航天航空ETF", "515980 人工智能ETF"],
        crowded_themes={"新能源车": 2.1, "半导体": 1.5}
    )
    
    test_portfolio = Portfolio(
        stocks=[
            {"etf_code": "159208", "weight": 0.6},
            {"etf_code": "515980", "weight": 0.4}
        ],
        bonds=[],
        commodities=[]
    )
    
    # 生成报告
    report = generate_monthly_report(
        target_date="2025年12月",
        macro_state=test_macro_state,
        policy_signal=test_policy_signal,
        market_cond=test_market_cond,
        portfolio=test_portfolio
    )
    
    print(report)
