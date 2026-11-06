"""
中间件 Middleware（02c）
============================================================
学习目标：
1. 理解中间件在 Agent 调用链里的位置：横切关注点，不侵入业务代码
2. 掌握 create_agent(middleware=[...]) 的基本用法
3. 验证点：摘要 / 限流 / 重试三种中间件分别「在哪个环节拦截、改变了什么行为」

先记住一句话：
    Callback    -> 「观察」：在旁边记录日志、统计耗时，不改变执行结果
    Middleware  -> 「拦截」：改写输入、拦截输出、决定是否重试，改变执行结果

对照 02h 从零写完整 Agent：那些散落的限次、重试、摘要逻辑，
中间件把它们统一收敛为可组合的钩子，在模型调用 / 工具执行前后插入。

   用户输入 → [中间件链] → 模型调用 → [中间件链] → 工具执行 → [中间件链] → 结果

运行：
    uv run python 02c_middleware.py
"""

import os

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent  # 组装 agent 的入口
from langchain.agents.middleware import (
    SummarizationMiddleware,   # 摘要（需真实 model）
    ModelCallLimitMiddleware,  # 限流
    ToolRetryMiddleware,       # 重试
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 注：LangChain 阶段统一使用 langchain-openai（ChatOpenAI），不再直接使用 openai 包。
#     SummarizationMiddleware 需要真实 model 生成摘要；
#     限流 / 重试中间件无需 model，可直接实例化。


# ============================================================
# 0. 初始化模型（读 .env，对齐 02h 风格）
# ============================================================

env_path = find_dotenv()
load_dotenv(env_path)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-6-terra"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ============================================================
# 1. 定义工具（复用 02a 的 @tool）
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    data = {"北京": "晴天，25℃", "上海": "多云，22℃", "深圳": "小雨，28℃"}
    return data.get(city, f"未知城市：{city}")


@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式（仅支持 + - * / 和括号）。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# ============================================================
# 2. 验证点①：摘要中间件 —— 拦截「模型调用前」，压缩长上下文
# ============================================================

print("=" * 60)
print("验证点①：SummarizationMiddleware（长上下文自动摘要）")
print("=" * 60)

# SummarizationMiddleware 需要一个 model 来生成摘要（第一个参数必填），
# 用 langchain-openai 的 ChatOpenAI 实例（统一方案，不直接用 openai 包）：
summarizer = SummarizationMiddleware(
    model=model,                 # 用哪个模型做摘要（复用上方 .env 读到的 model）
    trigger=("tokens", 8000),    # 上下文达到 8000 token 时触发
    keep=("messages", 20),       # 摘要后保留最近 20 条消息
)
print(f"已声明 {summarizer.name}：长对话自动把早期消息压成摘要")
print("（依赖 langchain-openai + 真实 key，故本脚本仅声明、不真正运行）")
print()


# ============================================================
# 3. 验证点②：限流中间件 —— 拦截「模型调用次数」，防止死循环
# ============================================================

print("=" * 60)
print("验证点②：ModelCallLimitMiddleware（限制模型调用次数）")
print("=" * 60)

# 限制整个会话里 LLM 被调用的次数，超过即终止，避免工具调用死循环。
limit = ModelCallLimitMiddleware(
    thread_limit=3,          # 单会话最多调用 3 次模型
    exit_behavior="error",   # 超限后抛错（另一种是 'end' 正常结束）
)
print(f"已声明 {limit.name}：单会话最多调用 3 次模型，防死循环")
print()


# ============================================================
# 4. 验证点③：重试中间件 —— 拦截「工具执行失败」，自动重试
# ============================================================

print("=" * 60)
print("验证点③：ToolRetryMiddleware（工具失败自动重试）")
print("=" * 60)

retry = ToolRetryMiddleware(
    max_retries=2,                  # 工具执行失败最多重试 2 次
    backoff_factor=2.0,             # 指数退避：1s -> 2s
    initial_delay=1.0,              # 首次重试前等待 1 秒
)
print(f"已声明 {retry.name}：工具报错自动重试，最多 2 次")
print()


# ============================================================
# 5. 组合多个中间件 —— 验证执行顺序（洋葱模型）
# ============================================================

print("=" * 60)
print("组合多个中间件（洋葱模型：列表顺序 = 执行顺序）")
print("=" * 60)

# 真实组装：把中间件列表传给 create_agent 的 middleware 参数
# （这里仅演示「怎么传」，不真正运行，因需要真实 model）
#   agent = create_agent(model, tools, middleware=[
#       SummarizationMiddleware(model=model),   # 第 1 层：先摘要
#       ModelCallLimitMiddleware(thread_limit=5),  # 第 2 层：再限流
#       ToolRetryMiddleware(max_retries=2),        # 第 3 层：最后重试
#   ])
print("""
    请求进来
      → SummarizationMiddleware 前置
        → ModelCallLimitMiddleware 前置
          → ToolRetryMiddleware 前置
            → 模型 / 工具核心逻辑
          → ToolRetryMiddleware 后置
        → ModelCallLimitMiddleware 后置
      → SummarizationMiddleware 后置
    响应出去

    顺序即语义：先摘要还是先限流，行为不同。
    挂载方式：create_agent(model, tools, middleware=middleware_chain)
""")


# ============================================================
# 6. 验证点小结
# ============================================================

print("=" * 60)
print("验证点小结：三种中间件的拦截环节与行为变化")
print("=" * 60)
print("""
    ┌──────────────────────────────┬─────────────────┬───────────────────────────┐
    │ 中间件                        │ 拦截环节         │ 改变的行为                 │
    ├──────────────────────────────┼─────────────────┼───────────────────────────┤
    │ SummarizationMiddleware      │ 模型调用前       │ 长上下文压成摘要，防 token 爆 │
    │ ModelCallLimitMiddleware     │ 模型调用次数     │ 超过上限强制终止，防死循环   │
    │ ToolRetryMiddleware          │ 工具执行失败     │ 自动重试，最多 max_retries 次 │
    └──────────────────────────────┴─────────────────┴───────────────────────────┘

    一句话：Callback 是「看」，Middleware 是「管」。
    想记录发生了什么 -> Callback（02g）；想改变接下来发生什么 -> Middleware（02c）。
""")
