"""
从零写一个完整带工具调用的 Agent（Day 3 · 达标验收）
============================================================
验收标准：
1. 带 ≥2 个工具（本文件提供 3 个：天气 / 计算器 / 时间）
2. 带记忆（Window 滑动窗口，复用 Day 2 思路）
3. 带 Callback（复用 Day 2 思路，替代 print）
4. 能独立跑通多轮对话 + 多工具协作

本文件综合 Day 1（@tool + StructuredTool）+ Day 2（Memory + Callback），
用 LangChain 真实组件组装，但核心循环逻辑保持 mini-agent 的手写风格，
让你看清框架「帮你做了什么、你还需要自己做什么」。

运行：
    python 05_agent.py
"""

import os
import json
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# 1. 初始化配置（同 mini-agent）
# ============================================================

env_path = find_dotenv()
load_dotenv(env_path)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-terra")

if not API_KEY or not BASE_URL:
    raise ValueError("缺少 OPENAI_API_KEY / OPENAI_BASE_URL，请检查 .env 文件")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================================
# 2. 定义工具（Day 1 知识：@tool 快速定义 + StructuredTool 强校验）
# ============================================================

def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    data = {"北京": "晴天，25℃", "上海": "多云，22℃", "深圳": "小雨，28℃"}
    return data.get(city, f"未知城市：{city}")


class CalcInput(BaseModel):
    expression: str = Field(description="需要计算的数学表达式，如 '1+2*3'")


def calculate(expression: str) -> str:
    """计算一个数学表达式（仅支持 + - * / 和括号）。"""
    try:
        # 安全起见，这里用 eval 仅演示；真实项目应使用受限求值器
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 用 StructuredTool 组装（演示 Day 1 的两种定义方式混用）
tools = [
    StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="查询指定城市的实时天气",
    ),
    StructuredTool.from_function(
        func=calculate,
        name="calculate",
        description="计算数学表达式",
        args_schema=CalcInput,
    ),
    StructuredTool.from_function(
        func=get_current_time,
        name="get_current_time",
        description="获取当前日期和时间",
    ),
]

# 转成 OpenAI function calling 需要的 tools 格式
tools_openai = []
for t in tools:
    tools_openai.append({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema.model_json_schema(),
        },
    })


# ============================================================
# 3. 记忆（Day 2 知识：Window 滑动窗口）
# ============================================================

class WindowMemory:
    """滑动窗口记忆：只保留最近 K 轮对话。"""

    def __init__(self, k: int = 8):
        self._messages = []
        self.k = k

    def add(self, message: dict) -> None:
        self._messages.append(message)
        self._messages = self._messages[-self.k:]

    def get_messages(self):
        return list(self._messages)


# ============================================================
# 4. Callback（Day 2 知识：替代 print）
# ============================================================

class AgentCallback:
    """统一的事件回调，负责所有可观测性输出。"""

    def on_tool_start(self, name: str, args: dict) -> None:
        print(f"  [Tool] 调用 {name}{args}")

    def on_tool_end(self, name: str, result: str) -> None:
        print(f"  [Tool] {name} -> {result}")

    def on_finish(self, answer: str) -> None:
        print(f"\n[Agent] {answer}")


# ============================================================
# 5. Agent 主循环（核心逻辑同 mini-agent，但工具可扩展 + 有记忆）
# ============================================================

def run_agent(question: str, memory: WindowMemory, callback: AgentCallback) -> str:
    # 记忆里已有历史，则把历史拼上（实现「多轮对话」）
    messages = memory.get_messages() + [{"role": "user", "content": question}]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_openai,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            # 生成最终回答，写入记忆并返回
            memory.add({"role": "user", "content": question})
            memory.add({"role": "assistant", "content": msg.content})
            callback.on_finish(msg.content)
            return msg.content

        for call in msg.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)

            callback.on_tool_start(tool_name, args)

            # 根据工具名分发执行
            tool_map = {t.name: t for t in tools}
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(args)
            else:
                result = "未知工具"

            callback.on_tool_end(tool_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })


# ============================================================
# 6. 验收入口：多轮对话 + 多工具协作
# ============================================================

if __name__ == "__main__":
    memory = WindowMemory(k=8)
    callback = AgentCallback()

    print("=" * 60)
    print("Day 3 验收：多轮对话 + 多工具协作")
    print("=" * 60)

    # 第 1 轮：单工具（天气）
    print("\n[User] 北京今天天气怎么样？")
    run_agent("北京今天天气怎么样？", memory, callback)

    # 第 2 轮：多工具协作（时间 + 计算）
    print("\n[User] 现在几点了？然后帮我算一下 (25 + 15) * 2 等于多少")
    run_agent("现在几点了？然后帮我算一下 (25 + 15) * 2 等于多少", memory, callback)

    # 第 3 轮：验证记忆（引用前文，无需重复提问）
    print("\n[User] 那我刚才问的第一个城市，天气适合出门吗？")
    run_agent("那我刚才问的第一个城市，天气适合出门吗？", memory, callback)
