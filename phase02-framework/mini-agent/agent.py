"""
最小 Agent：基于 Function Calling 实现天气查询。

核心思想：
    Agent = LLM + Tools + Loop

执行流程：
    1. 接收用户问题
    2. 将问题和工具定义发送给大模型
    3. 大模型决定是否调用工具
    4. Python 执行对应工具
    5. 将工具执行结果返回给大模型
    6. 大模型生成最终回答

运行：
    python agent.py
"""

import os
import json

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


# ============================================================
# 1. 初始化配置
# ============================================================

# 自动向上查找项目根目录下的 .env 文件
env_path = find_dotenv()
load_dotenv(env_path)

# 读取 API 配置
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")

# 校验必要配置
if not API_KEY:
    raise ValueError("缺少 OPENAI_API_KEY，请检查 .env 文件")

if not BASE_URL:
    raise ValueError("缺少 OPENAI_BASE_URL，请检查 .env 文件")

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================
# 2. 定义工具
# ============================================================

def get_weather(city: str) -> str:
    """
    查询指定城市的天气。

    Args:
        city: 城市名称

    Returns:
        str: 天气查询结果
    """

    # 模拟天气 API 返回，暂不接入真实天气服务
    return f"{city}：晴天，22°C"


# ============================================================
# 3. 注册工具
# ============================================================

# 将 Python 函数描述为模型可以识别的工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如成都、北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


# ============================================================
# 4. Agent 主循环
# ============================================================

def run_agent(question: str) -> str:
    """
    执行 Agent 思考与工具调用循环。

    Args:
        question: 用户输入的问题

    Returns:
        str: 大模型生成的最终回答
    """

    # 初始化对话上下文
    messages = [
        {"role": "user", "content": question}
    ]

    # 持续与模型交互，直到获得最终回答
    while True:

        # 4.1 将对话上下文和工具定义发送给模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        # 获取模型本轮回复
        msg = response.choices[0].message

        # 将模型回复追加到对话上下文
        messages.append(msg)

        # 4.2 判断模型是否需要调用工具
        # 没有工具调用，说明模型已经生成最终回答
        if not msg.tool_calls:
            return msg.content

        # 4.3 执行模型指定的工具
        for call in msg.tool_calls:

            # 获取工具名称
            tool_name = call.function.name

            # 解析模型生成的 JSON 参数
            args = json.loads(call.function.arguments)

            # 根据工具名称执行对应的 Python 函数
            if tool_name == "get_weather":
                result = get_weather(**args)
            else:
                result = "未知工具"

            # 打印工具调用过程，方便调试
            print(f"\n[Tool] {tool_name}({args})")
            print(f"[Result] {result}")

            # 4.4 将工具执行结果返回给模型
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })


# ============================================================
# 5. 程序入口
# ============================================================

if __name__ == "__main__":

    # 用户输入
    question = "成都今天天气怎么样？"

    print(f"[User] {question}")

    # 启动 Agent
    answer = run_agent(question)

    # 输出最终回答
    print(f"\n[Agent] {answer}")