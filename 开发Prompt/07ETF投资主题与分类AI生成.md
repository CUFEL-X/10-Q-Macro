# ETF投资主题分类系统 — LangChain智能标签引擎开发方案

## 方案概述

本方案旨在使用 LangChain 构建轻量级智能标签引擎，对 ETF 基础数据进行自动化主题分类与标签生成。通过大语言模型（LLM）的语义理解能力，实现对 ETF 投资主题的精准识别与分类，为 Q-Macro 策略系统提供结构化的 ETF 主题数据。

### 核心目标
- 使用 LangChain 构建轻量级智能标签引擎，对 `etf_basic.csv` 中的每只 ETF 进行主题分类
- 生成标准化的 ETF 标签数据，包括投资主题、资产类别和宏观敏感性
- 确保标签系统的准确性、一致性和可扩展性

### 设计原则
- **简洁可靠**：仅使用 LangChain 基础组件，避免过度抽象
- **结构化输出**：采用 Pydantic 模型确保输出格式规范统一
- **容错处理**：实现重试机制，确保批处理过程稳定可靠
- **可中断续跑**：支持增量更新，提高处理效率

### 环境配置
- API Key 已在 `Q-Macro/.env` 文件中配置
- 支持多种大语言模型，包括 Qwen、DeepSeek 等兼容 OpenAI 接口的模型

---

## 文件路径

- **标签引擎代码**：`Q-Macro/src/agents/theme_mapper.py`
- **输入数据**：`Q-Macro/data/etf/etf_basic.csv`
- **输出数据**：`Q-Macro/data/etf/processed_etf_basic.csv`

---

## 技术栈选择

| 需求 | LangChain 组件 | 选择理由 |
|------|----------------|----------|
| 大语言模型调用 | `ChatOpenAI`（兼容 Qwen/DeepSeek/OpenRouter） | 支持从 `.env` 文件自动加载 API Key 和基础 URL |
| 结构化输出处理 | `PydanticOutputParser` | 强制 JSON Schema 格式，减少模型输出幻觉 |
| 提示词管理 | `ChatPromptTemplate` | 清晰分离指令与变量，提高提示词可维护性 |
| 批处理机制 | 普通 for 循环 + 重试逻辑 | 无需复杂链结构，实现简单可靠的批处理 |

> **技术选择说明**：本任务为"单步推理"类型，无需使用 Agent Executor 或 Tools 组件，仅通过基础组件即可满足需求。

---

## 依赖配置

在 `pyproject.toml` 文件中添加以下依赖项：

```toml
[tool.uv.dependencies]
langchain = "*"
langchain-openai = "*"  # 支持通过 base_url 配置使用 Qwen/DeepSeek 等模型
pydantic = ">=2.0"  # 用于结构化输出验证
python-dotenv = "*"  # 用于加载环境变量
```

> **使用说明**：LangChain 的 `ChatOpenAI` 组件可通过 `base_url` 参数支持任意兼容 OpenAI 接口的模型，如阿里百炼、OpenRouter 等。

---

## 核心组件设计

### 1. 标签输出模型

```python
# src/agents/theme_mapper.py

from pydantic import BaseModel, Field

class ETFTag(BaseModel):
    """ETF标签模型"""
    theme: str = Field(
        description="ETF投资主题，从预定义列表中选择",
        examples=["人工智能", "新能源", "消费"]
    )
    asset_class: str = Field(
        description="资产大类",
        examples=["股票", "债券", "商品", "货币"]
    )
    macro_sensitivity: str = Field(
        description="宏观经济敏感性",
        examples=["高增长敏感", "抗通胀", "避险"]
    )
```

### 2. 模型与解析器初始化

```python
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen-max"),
    temperature=0.0,  # 降低随机性，提高输出一致性
    max_retries=2  # 内置重试机制
)

# 初始化结构化输出解析器
parser = PydanticOutputParser(pydantic_object=ETFTag)
```

### 3. 提示词模板设计

```python
# 构建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的ETF分析师，请根据以下ETF信息，按照要求输出结构化的标签信息。"),
    ("user", """
【ETF信息】
- 中文简称: {csname}
- 跟踪指数名称: {index_name}
- 指数中文名: {indx_csname}

【输出规则】
{format_instructions}

【注意事项】
- 请严格从预定义选项中选择标签，不得自行编造
- 若无法判断合适的标签，请选择"其他"
- 只输出JSON格式的标签信息，不要包含其他文字
""")
])

# 注入格式指令
full_prompt = prompt.partial(format_instructions=parser.get_format_instructions())
```

### 4. 单只ETF标签生成函数

```python
import time

def tag_single_etf(etf_row: dict, max_retries: int = 2) -> dict:
    """
    对单只ETF进行标签生成，失败时返回默认值
    
    Args:
        etf_row: ETF基础信息字典
        max_retries: 最大重试次数
        
    Returns:
        dict: 包含theme、asset_class和macro_sensitivity的标签字典
    """
    for attempt in range(max_retries + 1):
        try:
            # 构建处理链
            chain = full_prompt | llm | parser
            # 执行标签生成
            result = chain.invoke({
                "csname": etf_row.get("csname", ""),
                "index_name": etf_row.get("index_name", ""),
                "indx_csname": etf_row.get("indx_csname", "")
            })
            # 返回字典格式结果
            return result.dict()
        except Exception as e:
            if attempt == max_retries:
                print(f"尝试{max_retries+1}次后失败: {e}")
                # 返回默认标签
                return {"theme": "其他", "asset_class": "其他", "macro_sensitivity": "其他"}
            # 简单退避后重试
            time.sleep(1)
```

### 5. 批量标签生成主函数

```python
import pandas as pd

def batch_tag_etfs(
    input_path: str = "data/etf/etf_basic.csv",
    output_path: str = "data/etf/processed_etf_basic.csv"
) -> None:
    """
    批量为ETF生成标签并保存结果
    
    Args:
        input_path: ETF基础数据文件路径
        output_path: 标签结果保存路径
    """
    # 读取输入数据
    df = pd.read_csv(input_path)
    
    # 检查是否存在已处理文件，支持增量更新
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        # 检查必要列是否存在
        required_columns = ["theme", "asset_class", "macro_sensitivity"]
        if all(col in existing_df.columns for col in required_columns):
            df = existing_df  # 继续处理未完成的任务
    
    # 确保标签列存在
    for col in ["theme", "asset_class", "macro_sensitivity"]:
        if col not in df.columns:
            df[col] = "其他"
    
    # 逐行处理ETF数据
    for idx, row in df.iterrows():
        # 跳过已处理的ETF
        if row["theme"] == "其他":
            # 生成标签
            tags = tag_single_etf(row.to_dict())
            # 更新标签信息
            df.at[idx, "theme"] = tags["theme"]
            df.at[idx, "asset_class"] = tags["asset_class"]
            df.at[idx, "macro_sensitivity"] = tags["macro_sensitivity"]
            # 打印处理状态
            print(f"已标记 ETF {row['code']}: {tags['theme']}")
            # 避免API限流
            time.sleep(0.1)
    
    # 保存处理结果
    df.to_csv(output_path, index=False)
    print(f"✅ 标签生成完成！结果已保存至: {output_path}")
```

---

## 环境配置示例

在 `Q-Macro/.env` 文件中添加以下配置：

```env
# 大语言模型 API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  # 阿里百炼
LLM_MODEL=qwen-max
```

> **配置说明**：LangChain 会自动读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 环境变量，无需在代码中硬编码。

---

## 使用方式

在脚本或交互式环境中调用标签引擎：

```python
# 导入批量标签生成函数
from src.agents.theme_mapper import batch_tag_etfs

# 执行批量标签生成
batch_tag_etfs()

# 或指定自定义路径
batch_tag_etfs(
    input_path="custom/path/to/etf_basic.csv",
    output_path="custom/path/to/processed_etf_basic.csv"
)
```

---

## 方案优势

| 特性 | 实现方式 | 优势 |
|------|----------|------|
| **简洁高效** | 仅使用 LangChain 基础三件套：Prompt + LLM + Parser | 减少复杂性，提高代码可维护性 |
| **输出可靠** | Pydantic 模型强制结构化输出，配合重试机制 | 确保标签数据格式规范，减少错误 |
| **可中断续跑** | 支持增量更新，跳过已处理的 ETF | 提高处理效率，支持分批处理 |
| **环境友好** | 自动读取 `.env` 配置，无需硬编码 API Key | 增强安全性，便于不同环境部署 |
| **策略就绪** | 输出格式与 `portfolio_builder.py` 直接兼容 | 无缝集成到投资组合构建流程 |
| **模型灵活** | 支持多种兼容 OpenAI 接口的大语言模型 | 可根据性能和成本需求选择合适模型 |

---

本方案完全遵循 LangChain 官方推荐的 "Structured Output" 最佳实践，实现了一个简单、可靠、高效的 ETF 智能标签引擎。通过大语言模型的语义理解能力，结合结构化输出验证，确保了 ETF 标签的准确性和一致性，为 Q-Macro 策略系统提供了高质量的主题分类数据支持。