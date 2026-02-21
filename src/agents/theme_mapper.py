"""ETF标签工程师 - 使用LangChain对ETF进行主题分类和打标签"""
from pydantic import BaseModel, Field, model_validator, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import pandas as pd
import time
from typing import Dict, Any, Literal

# 加载环境变量
load_dotenv()

# Pydantic 输出模型
class ETFTag(BaseModel):
    asset_class: Literal["股票", "债券", "商品", "货币", "其他"]
    theme: str  # 第二级分类：如宽基、行业主题等
    sub_theme: str  # 第三级分类：行业主题下的详细主题
    
    @model_validator(mode='after')
    def validate_theme_by_asset_class(self) -> 'ETFTag':
        valid_themes = {
            "股票": ["宽基", "主题"],
            "债券": ["利率债", "信用债", "可转债", "短债", "其他债券"],
            "商品": ["黄金", "白银", "原油", "农产品", "其他商品"],
            "货币": ["货币基金"],
            "其他": ["其他"]
        }
        if self.theme not in valid_themes[self.asset_class]:
            # 容忍invalid的分类，记录警告信息
            print(f"Warning: Invalid theme '{self.theme}' for asset_class '{self.asset_class}', using it anyway")
        return self

# 检查API密钥和基础URL
api_key = os.getenv("LLM_API_KEY")
api_base = os.getenv("LLM_API_BASE")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Next-80B-A3B-Instruct")

if not api_key:
    print("Warning: LLM_API_KEY is not set, using 'EMPTY' as default")
    api_key = "EMPTY"

if not api_base:
    raise ValueError("LLM_API_BASE not found in environment variables. Please set it in .env file.")

# 初始化 LLM
llm = ChatOpenAI(
    model=model_name,
    temperature=0.0,  # 降低随机性
    max_retries=2,
    api_key=api_key,
    base_url=api_base
)

# 初始化解析器
parser = PydanticOutputParser(pydantic_object=ETFTag)

# 创建 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的ETF分析师，请根据以下信息严格按要求输出JSON。"),
    ("user", """
    【ETF信息】
    - 中文简称: {csname}
    - 跟踪指数名称: {index_name}
    - 指数中文名: {indx_csname}

    【分类规则】

    1. 首先判断资产大类（asset_class）：
       - 股票：跟踪股票指数（如沪深300、人工智能）
       - 债券：名称含"国债""企业债""可转债"等
       - 商品：含"黄金""原油"等实物资产
       - 货币：货币基金类
       - 其他：无法归类

    2. 再根据资产大类选择主题（theme）：

       ▶ 股票类：
         - 宽基：跟踪各类宽基指数
         - 主题：跟踪行业、主题等指数

       ▶ 债券类：
         - 利率债：国债、政金债
         - 信用债：企业债、公司债
         - 可转债：可转换债券
         - 短债：短期限债券

       ▶ 商品类：直接匹配商品类型（黄金/原油等）

    3. 对于股票类，请进一步指定详细主题（sub_theme）：
       - 宽基的sub_theme：大盘宽基、中盘宽基、小盘宽基、全市场宽基等
       - 主题的sub_theme：人工智能、半导体、新能源车、光伏、医药、消费、金融、周期、高端制造、数字经济、国企改革、红利低波等
       - 对于其他资产大类，sub_theme可以与theme相同

    4. 输出要求：
       - 只输出一个JSON对象
       - 字段：{format_instructions}
       - 若不确定，asset_class="其他", theme="其他", sub_theme="其他"

    【输出示例】
    {{"asset_class": "股票", "theme": "宽基", "sub_theme": "大盘宽基"}}
    {{"asset_class": "股票", "theme": "主题", "sub_theme": "人工智能"}}
    {{"asset_class": "股票", "theme": "主题", "sub_theme": "国企改革"}}
    {{"asset_class": "商品", "theme": "黄金", "sub_theme": "黄金"}}
    """)
])

# 注入格式指令
full_prompt = prompt.partial(format_instructions=parser.get_format_instructions())

def tag_single_etf(etf_row: dict, max_retries: int = 2) -> dict:
    """对单只ETF打标签，失败返回默认值"""
    for attempt in range(max_retries + 1):
        try:
            chain = full_prompt | llm | parser
            result = chain.invoke({
                "csname": etf_row.get("csname", ""),
                "index_name": etf_row.get("index_name", ""),
                "indx_csname": etf_row.get("indx_csname", "")
            })
            return result.model_dump()
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed after {max_retries+1} attempts: {e}")
                # 解析错误信息，尝试提取asset_class
                error_str = str(e)
                if "asset_class" in error_str and "股票" in error_str:
                    return {"asset_class": "股票", "theme": "主题", "sub_theme": "其他"}
                elif "asset_class" in error_str and "债券" in error_str:
                    return {"asset_class": "债券", "theme": "其他债券", "sub_theme": "其他"}
                elif "asset_class" in error_str and "商品" in error_str:
                    return {"asset_class": "商品", "theme": "其他商品", "sub_theme": "其他"}
                elif "asset_class" in error_str and "货币" in error_str:
                    return {"asset_class": "货币", "theme": "货币基金", "sub_theme": "货币基金"}
                else:
                    return {"asset_class": "其他", "theme": "其他", "sub_theme": "其他"}
            time.sleep(1)  # 简单退避

def batch_tag_etfs(
    input_path: str = "data/etf/etf_basic.csv",
    output_path: str = "data/etf/processed_etf_basic.csv"
) -> None:
    """批量打标签并保存"""
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 读取ETF基本信息
    df = pd.read_csv(input_path)
    
    # 若已存在输出文件，尝试增量更新
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if all(col in existing.columns for col in ["theme", "asset_class", "sub_theme"]):
            df = existing  # 继续未完成的任务
            print("Found existing tagged data, continuing from where we left off...")
    
    # 确保新列存在
    for col in ["theme", "asset_class", "sub_theme"]:
        if col not in df.columns:
            df[col] = "其他"
    
    # 逐行处理
    total = len(df)
    for idx, row in df.iterrows():
        if row["theme"] == "其他":  # 跳过已打标签的
            print(f"Processing {idx+1}/{total}: {row['code']}")
            tags = tag_single_etf(row.to_dict())
            df.at[idx, "theme"] = tags["theme"]
            df.at[idx, "asset_class"] = tags["asset_class"]
            df.at[idx, "sub_theme"] = tags["sub_theme"]
            print(f"Tagged {row['code']}: {tags['asset_class']} | {tags['theme']} | {tags['sub_theme']}")
            time.sleep(0.1)  # 避免 API 限流
    
    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"✅ 标签完成！结果保存至: {output_path}")

def analyze_etf_tags(input_path: str = "data/etf/processed_etf_basic.csv", output_path: str = "data/etf/tag_analysis.txt") -> None:
    """分析ETF标签数据并生成描述性统计"""
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 读取数据
    df = pd.read_csv(input_path)
    
    # 确保标签列存在
    required_cols = ["asset_class", "theme", "sub_theme"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in input file")
    
    # 创建统计结果
    analysis = []
    analysis.append("ETF标签描述性统计分析\n")
    analysis.append("=" * 50 + "\n")
    
    # 1. 资产大类统计
    analysis.append("1. 资产大类分布\n")
    analysis.append("-" * 30 + "\n")
    asset_class_counts = df["asset_class"].value_counts()
    for asset, count in asset_class_counts.items():
        analysis.append(f"{asset}: {count}只 ({count/len(df)*100:.1f}%)\n")
    analysis.append("\n")
    
    # 2. 股票类详细统计
    stock_df = df[df["asset_class"] == "股票"]
    if not stock_df.empty:
        analysis.append("2. 股票类详细分布\n")
        analysis.append("-" * 30 + "\n")
        
        # 股票主题分布
        stock_theme_counts = stock_df["theme"].value_counts()
        analysis.append("2.1 股票主题分布\n")
        for theme, count in stock_theme_counts.items():
            analysis.append(f"{theme}: {count}只 ({count/len(stock_df)*100:.1f}%)\n")
        analysis.append("\n")
        
        # 宽基子主题分布
        broad_based_df = stock_df[stock_df["theme"] == "宽基"]
        if not broad_based_df.empty:
            analysis.append("2.2 宽基子主题分布\n")
            broad_based_subtheme_counts = broad_based_df["sub_theme"].value_counts()
            for sub_theme, count in broad_based_subtheme_counts.items():
                analysis.append(f"{sub_theme}: {count}只 ({count/len(broad_based_df)*100:.1f}%)\n")
            analysis.append("\n")
            
            # 宽基例子
            analysis.append("2.3 宽基ETF例子\n")
            for sub_theme in broad_based_subtheme_counts.index[:3]:  # 每个子主题取前3个例子
                sub_theme_examples = broad_based_df[broad_based_df["sub_theme"] == sub_theme].head(3)
                analysis.append(f"{sub_theme}:\n")
                for _, row in sub_theme_examples.iterrows():
                    analysis.append(f"  - {row['csname']} ({row['code']})\n")
            analysis.append("\n")
        
        # 主题子主题分布
        theme_df = stock_df[stock_df["theme"] == "主题"]
        if not theme_df.empty:
            analysis.append("2.4 主题子主题分布\n")
            theme_subtheme_counts = theme_df["sub_theme"].value_counts()
            for sub_theme, count in theme_subtheme_counts.items():
                analysis.append(f"{sub_theme}: {count}只 ({count/len(theme_df)*100:.1f}%)\n")
            analysis.append("\n")
            
            # 主题例子
            analysis.append("2.5 主题ETF例子\n")
            for sub_theme in theme_subtheme_counts.index[:5]:  # 取前5个主题的例子
                sub_theme_examples = theme_df[theme_df["sub_theme"] == sub_theme].head(3)
                analysis.append(f"{sub_theme}:\n")
                for _, row in sub_theme_examples.iterrows():
                    analysis.append(f"  - {row['csname']} ({row['code']})\n")
            analysis.append("\n")
    
    # 3. 其他资产类统计
    analysis.append("3. 其他资产类概览\n")
    analysis.append("-" * 30 + "\n")
    other_assets = df[df["asset_class"] != "股票"]["asset_class"].unique()
    for asset in other_assets:
        asset_df = df[df["asset_class"] == asset]
        analysis.append(f"{asset}: {len(asset_df)}只\n")
        # 取前3个例子
        examples = asset_df.head(3)
        for _, row in examples.iterrows():
            analysis.append(f"  - {row['csname']} ({row['code']})\n")
        analysis.append("\n")
    
    # 4. 总体统计
    analysis.append("4. 总体统计\n")
    analysis.append("-" * 30 + "\n")
    analysis.append(f"总ETF数量: {len(df)}只\n")
    analysis.append(f"标签覆盖率: {len(df[df['theme'] != '其他'])}/{len(df)} = {len(df[df['theme'] != '其他'])/len(df)*100:.1f}%\n")
    analysis.append(f"唯一子主题数量: {df['sub_theme'].nunique()}\n")
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(analysis)
    
    print(f"✅ 统计分析完成！结果保存至: {output_path}")

if __name__ == "__main__":
    # 测试批量打标签
    try:
        batch_tag_etfs()
        # 执行统计分析
        analyze_etf_tags()
    except Exception as e:
        print(f"Error: {e}")
